import gym
import numpy as np
import random
import gym.spaces

from gym.envs.registration import register

register(
    id='StagHuntEnv-v0',
    entry_point='spinup.test_envs.stag_hunt_env:StagHuntEnv'
)

class StagHuntEnv(gym.Env):
    """ 
    An environment inspired by the game "the stag hunt" that is well known in 
    game theory and is often interpreted as a good model for testing trust 
    and the ability to cooperate.
    This version of the game can even be played by a single agent who needs
    to "cooperate" with "her future self", so to speak - or, in other words,
    plan ahead and avoid getting trapped in a local optimum along the way.
    The game works as follows: Each episode lasts for n steps. At each step, the 
    agents decide to play cooperatively (hunt stag) or not (hunt hare). 
    An agent that hunts hare receives a small reward for it. Agents that
    hunt stag receive nothing for the first n-1 steps and then a large reward
    in the n-th step, if no single agent has played "hare" in the entire episode.
    """
    def __init__(self, sequence_length=5, n_agents=1):
        super(StagHuntEnv, self).__init__()
        self.sequence_length = sequence_length
        self.n_agents = n_agents
        self.obs_shape = (1,)
        self.action_space = [gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) for i in range(self.n_agents)]
        self.observation_space = [gym.spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape) for i in range(self.n_agents)]

        self.uniform_obs = np.zeros(self.obs_shape) # Let's make all states look the same, so that the agents really just need to trust each other (and themselves).
        self.stag_hunt_possible = True

        self.step_counter = 0

    def step(self, actions):
        for a in actions:
            if a[0] <= 0:
                self.stag_hunt_possible = False

        done = self.step_counter >= self.sequence_length - 1

        rewards = []
        for i in range(self.n_agents):
            if actions[i][0] <= 0:
                rewards.append(1.0 / self.sequence_length)
            elif done and self.stag_hunt_possible:
                rewards.append(2.0)
            else:
                rewards.append(0)
        self.step_counter += 1
        return [self.uniform_obs] * self.n_agents, rewards, [done] * self.n_agents, {}

    def reset(self):
        self.stag_hunt_possible = True
        self.step_counter = 0
        return [self.uniform_obs] * self.n_agents

    def render(self, mode="rgb_array"):
        pass       

