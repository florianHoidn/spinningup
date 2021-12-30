import gym
import numpy as np

class NormalizingEnv(gym.Wrapper):
    """
    An environment wrapper that normalizes incoming observations as well as outgoing actions.
    For the observations, the normalization is done dynamically by keeping track of filtered
    averages and stddevs for all features. For the actions vectors, the normalization is done
    statically on the basis of an initial estimate of what would be a sensible mean and stddev
    for the given game. 
    """
    def __init__(self, env, act_mean, act_var):
        super().__init__(env)
        self.act_mean = act_mean
        self.act_var = act_var
        env_obs_spaces = env.observation_space
        env_act_spaces = env.action_space
        self.single_agent_parent = False
        if not type(env_obs_spaces) is list:
            self.single_agent_parent = True
            env_obs_spaces = [env_obs_spaces]
            env_act_spaces = [env_act_spaces]
        if self.act_mean is not None and self.act_var is not None:
            if not type(act_mean) is list:
                act_mean = [np.copy(act_mean) for _ in range(len(env_act_spaces))]
            if not type(act_var) is list:
                act_var = [np.copy(act_var) for _ in range(len(env_act_spaces))]

        self.n_agents = len(env_obs_spaces)
        self.dict_space = False
        self.obs_normalizers = []
        for obs_space in env_obs_spaces:
            if isinstance(obs_space, gym.spaces.Dict):
                self.dict_space = True
                self.obs_normalizers.append({obs_key:AvgTracker(space.shape) for obs_key, space in obs_space.spaces.items()})
            else:
                self.obs_normalizers.append(AvgTracker(obs_space.shape))
        if self.act_mean is not None and self.act_var is not None:
            self.act_normalizers = []
            self.act_limits_low = []
            self.act_limits_high = []
            for i in range(self.n_agents):
                act_space = env_act_spaces[i]
                self.act_normalizers.append(AvgTracker(act_space.shape, init_mean=act_mean[i], init_var=act_var[i]))
                self.act_limits_low.append(np.array(act_space.low))
                self.act_limits_high.append(np.array(act_space.high))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.handle_obs(obs)

    def step(self, actions):
        if self.act_mean is None or self.act_var is None:
            rescaled_actions = actions
        else:
            rescaled_actions = self.handle_actions(actions)
        obs, rew, done, info = self.env.step(rescaled_actions)
        obs = self.handle_obs(obs)
        return obs, rew, done, info

    def handle_obs(self, obs):
        # TODO this might actually require clipping if there are any limits for the observations.
        if self.single_agent_parent:
            multi_obs = [obs]
        else:
            multi_obs = obs
        for i in range(self.n_agents):
            normalizers = self.obs_normalizers[i]
            if self.dict_space:
                for obs_key, normalizer in normalizers.items():
                    normalizer.update_normalizer(multi_obs[i][obs_key])
                    multi_obs[i][obs_key] = normalizer.normalize(multi_obs[i][obs_key])
            else:
                normalizers.update_normalizer(multi_obs[i])
                multi_obs[i] = normalizers.normalize(multi_obs[i])
        return multi_obs[0] if self.single_agent_parent else multi_obs

    def handle_actions(self, actions):
        if self.single_agent_parent:
            multi_actions = [actions]
        else:
            multi_actions = actions
        for i in range(self.n_agents):
            normalizers = self.act_normalizers[i]
            multi_actions[i] = np.clip(normalizers.undo_normalization(multi_actions[i]), self.act_limits_low[i], self.act_limits_high[i])
        return multi_actions[0] if self.single_agent_parent else multi_actions

class AvgTracker:
    def __init__(self, size, init_mean=None, init_var=None, avg_size=100000):
        self.avg = 0
        self.avg_counter = 0
        self.avg_size = avg_size
        self.filtered_mean = init_mean
        self.filtered_var = np.ones(size) if init_var is None else init_var
        self.filtered_stddev = np.ones(size) if init_var is None else np.sqrt(init_var)

    def update_normalizer(self, val):
        if self.filtered_mean is None:
            self.filtered_mean = val
        else:
            self.filtered_mean = (self.filtered_mean * self.avg_counter + val) / (self.avg_counter + 1)
        self.filtered_var = (self.filtered_var * self.avg_counter + (self.filtered_mean - val)**2) / (self.avg_counter + 1)
        if self.avg_counter < self.avg_size:
            self.avg_counter += 1

    def normalize(self, val):
        return ((val - self.filtered_mean) if self.filtered_mean is not None else val) / self.filtered_stddev

    def undo_normalization(self, val):
        scaled_val = val * self.filtered_stddev
        if self.filtered_mean is None:
            return scaled_val
        else:
            return scaled_val + self.filtered_mean