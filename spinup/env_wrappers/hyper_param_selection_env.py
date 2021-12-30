import gym
import numpy as np

class HyperParamSelectionEnv(gym.Wrapper):
    # This environment allows an agent to actively select a hyperparameter value at each step.
    def __init__(self, env, param_name, min_param_value, max_param_value):
        super().__init__(env)
        self.param_name = param_name
        self.min_param_value = min_param_value
        self.max_param_value = max_param_value
        self.mid_param_scale = (max_param_value - min_param_value) * 0.5
        self.mid_param_value = self.min_param_value + self.mid_param_scale
        self.current_param_value = self.mid_param_value
        if type(env.action_space) is list:
            self.multi_agent_parent = True
            action_spaces = []
            for space in env.action_space:
                low = np.append(-1.0, space.low * np.ones(space.shape)) if np.isscalar(space.low) else np.append(-1.0, space.low)
                high = np.append(1.0, space.high * np.ones(space.shape)) if np.isscalar(space.high) else np.append(1.0, space.high)
                action_spaces.append(gym.spaces.Box(low=low, high=high, dtype=space.dtype))
            self.action_space = action_spaces
        else:
            self.multi_agent_parent = False
            low = np.append(-1.0, env.action_space.low * np.ones(env.action_space.shape)) if np.isscalar(env.action_space.low) else np.append(-1.0, env.action_space.low)
            high = np.append(1.0, env.action_space.high * np.ones(env.action_space.shape)) if np.isscalar(env.action_space.high) else np.append(1.0, env.action_space.high)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=env.action_space.dtype)

    def reset(self, **kwargs):
        self.current_param_value = self.mid_param_value
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_param_value = self.mid_param_value + self.mid_param_scale * (action[:,0] if self.multi_agent_parent else action[0])
        return self.env.step(action[:,1:] if self.multi_agent_parent else action[1:])

    def get_param_value(self, param_name):
        if param_name == self.param_name:
            return self.current_param_value
        elif hasattr(super(), "get_param_value"):
            return super().get_param_value(param_name)
        return None