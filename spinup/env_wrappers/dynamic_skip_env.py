import gym
import numpy as np

class DynamicSkipEnv(gym.Wrapper):
    # This environment allows an agent to specify how often the chosen action should be repeated.
    def __init__(self, env, max_repetitions=10, render_steps=False):
        super().__init__(env)
        self.render_steps = render_steps
        self.skips_half = max_repetitions * 0.5
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
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action[:,1:] if self.multi_agent_parent else action[1:])
        total_rew = np.array(rew, dtype=np.float32) if self.multi_agent_parent else rew
        if not (any(done) if self.multi_agent_parent else done):
            for _ in range(int(self.skips_half + (self.skips_half * np.average(action[:,0] if self.multi_agent_parent else action[0])))): # TODO this doesn't really make sense in a multi agent evironment.
                obs, rew, done, info = self.env.step(action[:,1:] if self.multi_agent_parent else action[1:])
                total_rew += np.array(rew, dtype=np.float32) if self.multi_agent_parent else rew
                if self.render_steps:
                    self.render()
                if any(done) if self.multi_agent_parent else done:
                    break
        return obs, total_rew, done, info