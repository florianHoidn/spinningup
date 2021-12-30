import gym
import numpy as np

class ObsStackEnv(gym.Wrapper):
    # This environment stacks observations along the last axis to provide the agent with some temporal information.
    def __init__(self, env, stack_size, test_mode=False):
        super().__init__(env)
        self.test_mode = test_mode
        self.stack_size = stack_size
        env_obs_spaces = env.observation_space
        self.multi_agent_parent = type(env_obs_spaces) is list
        if not self.multi_agent_parent:
            env_obs_spaces = [env_obs_spaces]
        self.n_agents = len(env_obs_spaces)
        self.observation_space = []
        self.out_shape = []
        self.null_obs = []
        for i in range(self.n_agents):
            if isinstance(env_obs_spaces[i], gym.spaces.Dict):
                low = {k:np.tile(v, self.stack_size) for k,v in env_obs_spaces[i].low.items()}
                high = {k:np.tile(v, self.stack_size) for k,v in env_obs_spaces[i].high.items()}
                dtype = {k:np.tile(v, self.stack_size) for k,v in env_obs_spaces[i].dtype.items()}
                self.observation_space.append(gym.spaces.Dict(low=low, high=high, dtype=dtype))
                self.out_shape.append({k:list(v.shape[:-1]) + [v.shape[-1]*self.stack_size] for k,v in env_obs_spaces[i].items()})
                self.null_obs.append({k:np.zeros(shape=v.shape, dtype=v.dtype) for k,v in env_obs_spaces[i].items()})
            else:
                self.observation_space.append(gym.spaces.Box(low=np.tile(env_obs_spaces[i].low, self.stack_size), high=np.tile(env_obs_spaces[i].high, self.stack_size), dtype=env_obs_spaces[i].dtype))
                self.out_shape.append(list(env_obs_spaces[i].shape[:-1]) + [env_obs_spaces[i].shape[-1]*self.stack_size])
                self.null_obs.append(np.zeros(shape=env_obs_spaces[i].shape, dtype=env_obs_spaces[i].dtype))
        self.obs_stack = [[self.null_obs[i]] * self.stack_size for i in range(self.n_agents)]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.multi_agent_parent:
            obs = [obs]
        obs_combined = []
        self.obs_stack = []
        for i in range(self.n_agents):
            obs_stack_i = [self.null_obs[i]] * (self.stack_size-1)
            obs_stack_i.append(obs[i])
            self.obs_stack.append(obs_stack_i)
            if isinstance(obs[i], gym.spaces.Dict):
                obs_combined.append({k:np.reshape(np.concatenate([o[k] for o in obs_stack_i], axis=-1), newshape=self.out_shape[i][k]) for k in obs[i]})
            else:
                obs_combined.append(np.reshape(np.concatenate(obs_stack_i, axis=-1), newshape=self.out_shape[i]))
        return obs_combined

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if not self.multi_agent_parent:
            obs = [obs]
        obs_combined = []
        for i in range(self.n_agents):
            self.obs_stack[i].pop(0)
            self.obs_stack[i].append(obs[i])
            if isinstance(obs[i], gym.spaces.Dict):
                obs_combined.append({k:np.reshape(np.concatenate([o[k] for o in self.obs_stack[i]], axis=-1), newshape=self.out_shape[i][k]) for k in obs[i]})
            else:
                obs_combined.append(np.reshape(np.concatenate(self.obs_stack[i], axis=-1), newshape=self.out_shape[i]))
        return obs_combined, rew, done, info
