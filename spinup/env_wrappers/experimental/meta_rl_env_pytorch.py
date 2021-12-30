import gym
import numpy as np
import torch
from torch import nn

class MetaRlEnv(gym.Wrapper):

    # TODO not sure if I want to keep this.
    
    # An environment wrapper that interprets the incoming continuous action space vectors as (i) a numerical value that specifies the desired number of evaluation steps, followed by (ii) a list of weights for a mesa policy - i.e. the policy that the meta policy optimizes.
    # The specified weights will be loaded into the predefined model and evaluated in the underlaying environment for the specified number of steps.
    # Sounds expensive and it is - but also kinda cool :)
    def __init__(self, env, model, model_kw_args, cuda_device, render_steps=True, max_repetitions=200):
        super().__init__(env)

        self.cuda_device = cuda_device
        self.render_steps = render_steps
        self.skips_half = max_repetitions/2

        # Step 1: Build the mesa policy
        self.env_obs_dim = env.observation_space.shape[0]
        self.env_act_dim = env.action_space.shape[0]
        self.env_act_limit_low = env.action_space.low
        self.env_act_limit_high = env.action_space.high

        model_kw_args["input_sizes"] = [self.env_obs_dim] + model_kw_args["input_sizes"]
        model_kw_args["output_sizes"] += [self.env_act_dim]
        model_kw_args["output_activation"] = nn.Tanh
        self.model = model(**model_kw_args)
        if cuda_device is not None:
            self.model.to(cuda_device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Step 2: Build the meta action space.
        self.meta_act_dim = sum([np.prod(p.shape) for p in self.model.parameters()])
        abs_env_limit = np.max(np.abs(np.concatenate([self.env_act_limit_low, self.env_act_limit_high]))) # Let's make sure that the meta action space limits always allow the model to produce any unerlying action from bias alone. 
        low = np.append(-1.0, -abs_env_limit * np.ones(self.meta_act_dim))
        high = np.append(1.0, abs_env_limit * np.ones(self.meta_act_dim))
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_final_obs = obs
        return obs

    def step(self, action):
        # Step 1: load the chosen weights into the model.
        weights = action[1:]

        from_slice = 0
        for p in self.model.parameters():
            to_slice = from_slice + np.prod(p.shape)
            p.data.copy_(torch.reshape(torch.as_tensor(weights[from_slice:to_slice], dtype=torch.float32, device="cuda:0"), shape=p.shape))
            from_slice = to_slice

        # Step 2: evaluate the new model.
        eval_steps = int(self.skips_half + (self.skips_half * action[0]))
        obs, rew, done, info = self.env.step(self.model(torch.as_tensor(self.prev_final_obs, dtype=torch.float32, device=self.cuda_device)).cpu().numpy())
        if self.render_steps:
            self.env.render()
        total_rew = rew
        if not done:
            for _ in range(eval_steps):
                obs, rew, done, info = self.env.step(self.model(torch.as_tensor(obs, dtype=torch.float32, device=self.cuda_device)).cpu().numpy())
                if self.render_steps:
                    self.env.render()
                total_rew += rew
                if done:
                    break
        self.prev_final_obs = obs
        return obs, total_rew, done, info