import gym
import numpy as np
import random
import os
import torch
from torch import nn

class ObservationBuffer:
    """
    A simple FIFO buffer for observations.
    """
    def __init__(self, obs_dim_in, size, cuda_device):
        self.cuda_device = cuda_device
        self.obs_modalities = None
        if np.isscalar(obs_dim_in):
            obs_dim = [obs_dim_in]
        elif isinstance(obs_dim_in, gym.spaces.Dict):
            obs_dim = {k:v.shape for k,v in obs_dim_in.spaces.items()}
            self.obs_modalities = obs_dim.keys()
        else:
            obs_dim = obs_dim_in.shape
        self.obs_buf = {k:np.zeros((size,) + v, dtype=np.float32) for k,v in obs_dim.items()} if self.obs_modalities else np.zeros((size,) + obs_dim, dtype=np.float32)
        self.cur_idxs = []
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs):
        if self.obs_modalities:
            for k in self.obs_modalities:
                self.obs_buf[k][self.ptr] = obs[k]
        else:
            self.obs_buf[self.ptr] = obs
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        self.cur_idxs = np.random.randint(0, self.size, size=batch_size)
        return {k:torch.as_tensor(self.obs_buf[k][self.cur_idxs], dtype=torch.float32, device=self.cuda_device) for k in self.obs_buf} if self.obs_modalities else torch.as_tensor(self.obs_buf[self.cur_idxs], dtype=torch.float32, device=self.cuda_device)

class RndCuriosityEnv(gym.Wrapper):
    # This adds a random network distillation based curiosity bonus to the reward signal.
    def __init__(self, env, intrinsic_reward_weight, cuda_device, model_fn, model_kwargs_getter, 
                restore_model_path, save_model_path, save_freq=10000, split_reward_streams=False,
                burn_in_steps=500, predictor_train_interval=1, predictor_batch_size=256,
                start_predictor_training=300, replay_buffer_size=10000, max_avg_size=1000, 
                compute_hidden_step_rewards=True, add_hidden_step_rewards=False):
        super().__init__(env)
        from torch.optim import Adam

        self.cuda_device = cuda_device
        
        self.model_path = save_model_path
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        self.save_freq = save_freq

        self.split_reward_streams = split_reward_streams
        self.multi_agent_parent = type(env.observation_space) is list
        self.n_agents = len(env.observation_space) if self.multi_agent_parent else 1
        # TODO it would be nice if I could pass in a gym.spaces.Dict key to control what modality the agent should be curious about.
        self.rnd_net = []
        self.prediction_net = []
        self.pred_optimizer = []
        for i in range(self.n_agents):
            obs_space = env.observation_space[i] if self.multi_agent_parent else env.observation_space
            
            if restore_model_path is None or not os.path.exists(restore_model_path):
                rnd_net_i = ObservationModel(obs_space, model_fn, model_kwargs_getter, noisy_linear_layers=False)
                prediction_net_i = ObservationModel(obs_space, model_fn, model_kwargs_getter)
            else:
                model_file_name_template = os.path.join(restore_model_path, os.path.split(self.model_path)[1])
                rnd_net_i = torch.load(model_file_name_template.format("random", i))
                prediction_net_i = torch.load(model_file_name_template.format("prediction", i))

            self.rnd_net.append(rnd_net_i)
            self.prediction_net.append(prediction_net_i)
            if self.cuda_device is not None:
                self.rnd_net[i].to(cuda_device)
                self.prediction_net[i].to(cuda_device)
            self.pred_optimizer.append(Adam(self.prediction_net[i].parameters(), lr=5e-5, weight_decay=1e-6))
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.max_avg_size = max_avg_size
        self.avg_size = 0
        self.mu_obs = None
        self.var_obs = None
        self.epsilon = 1e-6
        self.max_batch_reward = [1.0] * self.n_agents
        self.mean_batch_reward = [1.0] * self.n_agents
        self.mean_loss_thresh = 0.0

        # It's recommended to recompute the intrinsic rewards for each batch, 
        # but it can be useful (e.g. for debugging or hyperparameter tuning) to compute them for each step, too.
        self.compute_hidden_step_rewards = compute_hidden_step_rewards
        self.add_hidden_step_rewards = add_hidden_step_rewards # If this is false, the rewards won't be added to the per step rewards - allowing the agent to recompute them for each batch.
        self.hidden_step_rewards = None

        self.total_steps = 0
        self.start_predictor_training = start_predictor_training
        self.predictor_train_interval = predictor_train_interval
        self.steps_since_update = 0
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffers = []
        for i in range(self.n_agents):
            self.replay_buffers.append(ObservationBuffer(env.observation_space[i] if self.multi_agent_parent else env.observation_space, replay_buffer_size, cuda_device))
        self.predictor_batch_size = predictor_batch_size

        if burn_in_steps > 0:
            self.reset()
            for i in range(burn_in_steps):
                _, _, done, _ = self.step(np.array([self.action_space[i].sample() for i in range(self.n_agents)]) if self.multi_agent_parent else self.action_space.sample())
                if (any(done) if self.multi_agent_parent else done) or i == burn_in_steps-1:
                    self.reset()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        
        obs = obs if type(obs) is list else [obs]
        
        for i in range(self.n_agents):
            self.replay_buffers[i].store(obs[i])

        mu_obs_next = []
        var_obs_next = []
        for i in range(self.n_agents):
            modalities = self.rnd_net[i].modalities
            if modalities:
                mu_obs_next.append({k:((self.avg_size * self.mu_obs[i][k] + obs[i][k])/(self.avg_size + 1)) if self.avg_size > 0 else obs[i][k] for k in modalities})
                var_obs_next.append({k:((self.avg_size * self.var_obs[i][k] + (mu_obs_next[i][k] - obs[i][k])**2)/(self.avg_size + 1)) if self.avg_size > 0 else (mu_obs_next[i][k] - obs[i][k])**2 for k in modalities})
            else:
                mu_obs_next.append(((self.avg_size * self.mu_obs[i] + obs[i])/(self.avg_size + 1)) if self.avg_size > 0 else obs[i])
                var_obs_next.append(((self.avg_size * self.var_obs[i] + (self.mu_obs[i] - obs[i])**2)/(self.avg_size + 1)) if self.avg_size > 0 else (mu_obs_next[i] - obs[i])**2)
        self.mu_obs = mu_obs_next
        self.var_obs = var_obs_next

        self.total_steps += 1
        if self.avg_size < self.max_avg_size and self.max_avg_size > 0:
            self.avg_size += 1

        self.steps_since_update += 1
        if self.avg_size >= self.start_predictor_training \
            and self.steps_since_update >= self.predictor_train_interval \
            and np.max(self.mean_batch_reward) > self.mean_loss_thresh:
            self.steps_since_update = 0
            self.update_predictor()

        if self.total_steps % self.save_freq == 0:
            for i in range(self.n_agents):
                torch.save(self.rnd_net[i], self.model_path.format("random", i))
                torch.save(self.prediction_net[i], self.model_path.format("prediction", i))

        if self.compute_hidden_step_rewards:
            self.hidden_step_rewards = []
            for i in range(self.n_agents):
                modalities = self.rnd_net[i].modalities
                if modalities:
                    self.hidden_step_rewards.append(self.compute_intrinsic_rewards(i, {k:torch.unsqueeze(torch.as_tensor(obs[i][k], dtype=torch.float32, device=self.cuda_device), dim=0) for k in modalities}, update_mean=False).detach().cpu().numpy())
                else:
                    self.hidden_step_rewards.append(self.compute_intrinsic_rewards(i, torch.unsqueeze(torch.as_tensor(obs[i], dtype=torch.float32, device=self.cuda_device), dim=0), update_mean=False).detach().cpu().numpy())
            if self.add_hidden_step_rewards:
                int_rew = self.hidden_step_rewards if self.multi_agent_parent else self.hidden_step_rewards[0]
                if self.split_reward_streams:    
                    rew_return = (rew, int_rew)
                elif self.multi_agent_parent: 
                    rew_return = [rew[i] + int_rew[i] for i in range(self.n_agents)]
                else:
                    rew_return = rew[0] + int_rew[0]
                return obs, rew_return, done, info
        
        return obs, (rew, [0] * self.n_agents if self.multi_agent_parent else 0) if self.split_reward_streams else rew, done, info

    def update_predictor(self):
        for i in range(self.n_agents):
            batch = self.replay_buffers[i].sample_batch(self.predictor_batch_size)
            modalities = self.rnd_net[i].modalities
            if modalities is not None:
                mu_tensor = {k:torch.as_tensor(self.mu_obs[i][k], dtype=torch.float32, device=self.cuda_device) for k in modalities}
                var_tensor = {k:torch.as_tensor(self.var_obs[i][k], dtype=torch.float32, device=self.cuda_device) for k in modalities}
                obs_batch_tensor = {k:(batch[k] - mu_tensor[k])/(torch.sqrt(var_tensor[k]) + self.epsilon) for k in modalities}
            else:
                mu_tensor = torch.as_tensor(self.mu_obs[i], dtype=torch.float32, device=self.cuda_device)
                var_tensor = torch.as_tensor(self.var_obs[i], dtype=torch.float32, device=self.cuda_device)
                obs_batch_tensor = (batch - mu_tensor)/(torch.sqrt(var_tensor) + self.epsilon)
            with torch.no_grad():
                rnd_out = self.rnd_net[i](obs_batch_tensor)
            pred_out = self.prediction_net[i](obs_batch_tensor)
            self.pred_optimizer[i].zero_grad()
            pred_loss = ((rnd_out - pred_out)**2).mean()
            pred_loss.backward()
            self.pred_optimizer[i].step()

    def compute_intrinsic_rewards(self, agent_nbr, obs, update_mean=True):
        if self.total_steps <= 0:
            return 0
        modalities = self.rnd_net[agent_nbr].modalities
        if modalities is not None:
            mu_tensor = {k:torch.as_tensor(self.mu_obs[agent_nbr][k], dtype=torch.float32, device=self.cuda_device) for k in modalities}
            var_tensor = {k:torch.as_tensor(self.var_obs[agent_nbr][k], dtype=torch.float32, device=self.cuda_device) for k in modalities}
            obs_tensor = {k:(obs[k] - mu_tensor[k]) / (torch.sqrt(var_tensor[k]) + 1e-9) for k in modalities}
        else:
            mu_tensor = torch.as_tensor(self.mu_obs[agent_nbr], dtype=torch.float32, device=self.cuda_device)
            var_tensor = torch.as_tensor(self.var_obs[agent_nbr], dtype=torch.float32, device=self.cuda_device)
            obs_tensor = (obs - mu_tensor) / (torch.sqrt(var_tensor) + 1e-9)
        with torch.no_grad():
            rnd_out = self.rnd_net[agent_nbr](obs_tensor)
            pred_out = self.prediction_net[agent_nbr](obs_tensor)
            pred_loss = ((rnd_out - pred_out)**2).mean(axis=-1)
            
            if update_mean:
                self.mean_batch_reward[agent_nbr] = torch.mean(pred_loss).detach()
                self.max_batch_reward[agent_nbr] = torch.max(torch.abs(pred_loss)).detach()
            pred_loss -= self.mean_batch_reward[agent_nbr] # Let's just generate non-zero curiousity for transitions that produce errors higher than average.
            pred_loss /= self.max_batch_reward[agent_nbr] + 1e-9
        return torch.clamp(pred_loss[0], 0.0, 1.0) * self.intrinsic_reward_weight

    def get_hidden_step_rewards(self):
        hidden_step_rewards_parent = None
        if hasattr(self.env, "get_hidden_step_rewards"):
            hidden_step_rewards_parent = self.env.get_hidden_step_rewards()
        if hidden_step_rewards_parent is not None:
            if self.hidden_step_rewards is None:
                return hidden_step_rewards_parent
            return np.array(self.hidden_step_rewards) + hidden_step_rewards_parent
        return np.array(self.hidden_step_rewards) if self.hidden_step_rewards is not None else None 
        
class ObservationModel(nn.Module):
    """
    The models that drive our random network distillation.
    """
    def __init__(self, env_obs_spaces, model_fn, model_kwargs_getter, noisy_linear_layers=None):
        super().__init__()

        self.modalities = None
        if isinstance(env_obs_spaces, gym.spaces.Dict):
            obs_dim = {k:np.array(v.shape) for k,v in env_obs_spaces.spaces.items()}
            self.modalities = list(env_obs_spaces.spaces.keys())
        elif isinstance(env_obs_spaces, dict):
            obs_dim = {k:v for k,v in env_obs_spaces.items()}
            self.modalities = list(env_obs_spaces.keys())
        else:
            obs_dim = env_obs_spaces.shape
        model_kwargs = model_kwargs_getter()
        model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] =  model_kwargs["output_sizes"] + [64]
        model_kwargs["output_activation"] = nn.Identity
        if noisy_linear_layers is not None and "noisy_linear_layers" in model_kwargs:
            model_kwargs["noisy_linear_layers"] = noisy_linear_layers
        self.model = model_fn(**model_kwargs)
        
    def forward(self, obs):
        return self.model(obs)
