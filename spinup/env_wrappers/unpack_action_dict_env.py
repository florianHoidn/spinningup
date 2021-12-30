import gym
import numpy as np

class UnpackActionDictEnv(gym.Wrapper):
    """
    Simple wrapper simplifies action spaces of type gym.spaces.Dict if possible.
    Some gym environments like the Minecraft envs from minerl use multi modal spaces
    not just for the observations, but also for the actions - without there being 
    much reason for doing so. This wrapper allows us to work with these environment, too.
    """
    def __init__(self, env):
        super().__init__(env)
        env_act_spaces = env.action_space
        if not type(env_act_spaces) is list:
            env_act_spaces = [env_act_spaces]

        self.n_agents = len(env_act_spaces)
        self.action_space_key, self.action_space = [], []
        for space in env_act_spaces:
            if isinstance(space, gym.spaces.Dict) and len(space.spaces) == 1:
                k_a, v_a = list(space.spaces.items())[0]
                self.action_space_key.append(k_a)
                self.action_space.append(v_a)
            else:
                print("Warning: UnpackActionDictEnv doesn't know how to properly simplify action space " + str(action_space))
                self.action_space_key.append(None)

    def step(self, actions):
        return self.env.step([{self.action_space_key[i]:actions[i]} if self.action_space_key[i] != None else actions[i] for i in range(self.n_agents)])
