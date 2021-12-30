import gym
import numpy as np

class ActionFeedbackEnv(gym.Wrapper):
    # This environment appends the agents' own actions to the observations.
    def __init__(self, env):
        super().__init__(env)
        env_obs_spaces = env.observation_space
        env_act_spaces = env.action_space
        if not type(env_obs_spaces) is list:
            env_obs_spaces = [env_obs_spaces]
            env_act_spaces = [env_act_spaces]

        self.n_agents = len(env_obs_spaces)
        self.observation_space = []
        self.null_action = []
        for i in range(self.n_agents):
            if isinstance(env_obs_spaces[i], gym.spaces.Dict):
                # TODO this doesn't work. See externalized memory for how to do this.
                low = dict(env_obs_spaces[i].low)
                low["action_feedback"] = env_act_spaces[i].low
                high = dict(env_obs_spaces[i].high)
                high["action_feedback"] = env_act_spaces[i].high
                dtype = dict(env_obs_spaces[i].dtype)
                dtype["action_feedback"] = env_act_spaces[i].dtype
                self.observation_space.append(gym.spaces.Dict(low=low, high=high, dtype=dtype))
            else:
                self.observation_space.append(gym.spaces.Box(low=np.concatenate([env_obs_spaces[i].low, env_act_spaces[i].low]), high=np.concatenate([env_obs_spaces[i].high, env_act_spaces[i].high]), dtype=env_obs_spaces[i].dtype))
            self.null_action.append(np.zeros(shape=env_act_spaces[i].shape, dtype=env_act_spaces[i].dtype))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(self.observation_space[0], gym.spaces.Dict):
            obs_per_agent = []
            for i in range(self.n_agents):
                combined_obs = dict(obs[i])
                combined_obs["action_feedback"] = self.null_action[i]
                obs_per_agent.append(combined_obs)
            return obs_per_agent
        else:
            return np.concatenate([[obs[i], self.null_action[i]] for i in range(self.n_agents)])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if isinstance(self.observation_space[0], gym.spaces.Dict):
            obs_per_agent = []
            for i in range(self.n_agents):
                combined_obs = dict(obs[i])
                combined_obs["action_feedback"] = action[i]
                obs_per_agent.append(combined_obs)
            return obs_per_agent
        else:
            return np.concatenate([[obs[i], action[i]] for i in range(self.n_agents)]), rew, done, info
