import gym
import numpy as np

class SingleAgentEnv(gym.Wrapper):
    """
    Simple wrapper that turns a standard single agent gym env into a one player multi agent env like https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/environment.py. 
    Really just to avoid having to wrap and check stuff everywhere in the code.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = [env.observation_space]
        self.action_space = [env.action_space]

    def reset(self, **kwargs):
        return [self.env.reset(**kwargs)]

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions[0])
        return [obs], [rew], [done], {'n':[info]}
