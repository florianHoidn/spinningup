import gym
import numpy as np

class AmpEnv(gym.Wrapper):
    """
    This environment wrapper allows you to load custom reference 
    observations for an imitation bonus inspired by DeepMimic's adverserial motion priors (AMPs), 
    see https://github.com/xbpeng/DeepMimic.
    While DeepMimic uses these reference observations to create high quality animations, the same
    idea can also be used to imitate policies in general.

    Args:
        env: The gym environment that is wrapped.
        demonstration_loader: An object that implements next_sequence, a method that loads a sequence of reference observations.
    """
    def __init__(self, env, demonstration_loader):
        super().__init__(env)

        # TODO maybe allow for different obs shapes per player.
        observation_space = env.observation_space[0] if isinstance(env.observation_space, list) else env.observation_space

        #if isinstance(observation_space, gym.spaces.Dict):
        #    self.amp_obs_size = {k:np.array(v.shape) for k,v in observation_space.spaces.items()}
        #else:
        #    self.amp_obs_size = observation_space.shape[0]
        
        #self.amp_obs_size = {"action":np.array(env.action_space[0].shape), "pov":np.array(observation_space.spaces["pov"].shape)}
        self.amp_obs_size = {"pov":np.array(observation_space.spaces["pov"].shape)}

        self.demonstration_loader = demonstration_loader
        self.prev_obs = None
        self.next_obs = None
        self.agent_act = None
        self.prev_state_amp_agent = {}
        self.expert_obs_idx = 0
        self.expert_sequence = self.demonstration_loader.next_sequence()

    def reset(self, **kwargs):
        self.next_obs = self.env.reset(**kwargs)
        self.agent_act = None
        return self.next_obs

    def step(self, actions):
        self.agent_act = actions
        self.prev_obs = self.next_obs
        self.next_obs, rew, done, info = self.env.step(actions)
        # TODO it can be super useful to see a plot of the current amp reward.
        return self.next_obs, rew, done, info

    def get_amp_obs(self, i):
        if self.agent_act is None:
            del self.prev_state_amp_agent[i]
            return None
        #amp_obs = {"state_amp_agent":self.agent_obs[i], "state_amp_expert":self.expert_sequence[self.expert_obs_idx]}
        #amp_obs = {"state_amp_agent":{"action":self.agent_act[i], "pov":self.prev_obs[i]["pov"]}, "state_amp_expert":self.expert_sequence[self.expert_obs_idx]}
        amp_obs = {"state_amp_agent":{"pov":self.prev_obs[i]["pov"]}, "state_amp_expert":self.expert_sequence[self.expert_obs_idx]}
        self.prev_state_amp_agent[i] = {k:[v] for k,v in amp_obs["state_amp_agent"].items()} # TODO mainly for debugging
        self.expert_obs_idx += 1
        if self.expert_obs_idx >= len(self.expert_sequence):
            self.expert_obs_idx = 0
            self.expert_sequence = self.demonstration_loader.next_sequence()

        amp_obs_stack = []
        if hasattr(self.env, "get_amp_obs"):
            amp_obs_stack = self.env.get_amp_obs(i)
        return amp_obs_stack + [amp_obs]

    def get_prev_state_amp_agent(self, i):
        # TODO mainly for debugging
        return self.prev_state_amp_agent[i] if i in self.prev_state_amp_agent else None

    def get_amp_obs_size(self):
        return self.amp_obs_size

    def stack_amp_replay_buffers(self, replay_buffer, model, imitation_reward_weight, restore_model_paths, agent_nbr, logger, cuda_device):
        if hasattr(self.env, "stack_amp_replay_buffers"):
            replay_buffer = self.env.stack_amp_replay_buffers(replay_buffer, model, imitation_reward_weight, restore_model_paths[:-1] if restore_model_paths is not None else None, agent_nbr, logger, cuda_device)

        from spinup.replay_buffers.amp_replay_buffer_pytorch import AMPReplayBuffer
        return AMPReplayBuffer(inner_replay_buffer=replay_buffer, 
            discriminator_model_fn=model.net, discriminator_model_kwargs_getter=model.get_tiny_kwargs,
            restore_model_path=restore_model_paths[-1] if restore_model_paths is not None and len(restore_model_paths) >= 1 else None,
            save_model_path=os.path.join(logger.output_dir, "pyt_save", "amp_discriminator_model_" + str(agent_nbr) + "_" + self.__class__.__name__ + ".pt"),
            imitation_reward_weight=imitation_reward_weight, amp_env=self, logger=logger, cuda_device=cuda_device)
