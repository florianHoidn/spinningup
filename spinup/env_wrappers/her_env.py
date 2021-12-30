import gym
import numpy as np

class GoalGeneratorEnv(gym.Wrapper):
    # This environment accompanies the hindsight replay env and allows you to use observations produced somewhere in the stack of environment wrappers to be used as goals in the hindsight replay.
    # The idea is to save resources by, e.g., not using entire observation stacks as goals.
    def __init__(self, env):
        super().__init__(env)
        env_obs_spaces = env.observation_space
        if not type(env_obs_spaces) is list:
            env_obs_spaces = [env_obs_spaces]
        n_agents = len(env_obs_spaces)
        self.prev_obs = [np.zeros(env_obs_spaces[i].shape, dtype=env_obs_spaces[i].dtype) for i in range(n_agents)]

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        return self.prev_obs

    def step(self, action):
        self.prev_obs, rew, done, info = self.env.step(action)
        return self.prev_obs, rew, done, info

    def get_goal(self):
        return self.prev_obs

class HerEnv(gym.Wrapper):
    # This environment concatenates a goal state to the observations and allows for hindsight experience replay style learing.
    # As the current goal for the on policy agent, the environment will automatically select the state with the largest accumulated reward so far.
    def __init__(self, env, k, her_reward, goal_generator_env, her_sample_strategy="future", test_mode=False, split_reward_streams=False):
        # TODO implement all her sample strategies - and make sure the implementation is clean (right now, it is a bit idiosyncratic - might still work, though).
        super().__init__(env)
        self.goal_generator_env = goal_generator_env
        self.test_mode = test_mode

        env_obs_spaces = env.observation_space
        if not type(env_obs_spaces) is list:
            env_obs_spaces = [env_obs_spaces]
        self.dict_space = False
        self.n_agents = len(env_obs_spaces)
        self.observation_space = []
        for i in range(self.n_agents):
            # TODO should I cover the case where only one of the envs is a Dict space? 
            if isinstance(env_obs_spaces[i], gym.spaces.Dict) and isinstance(self.goal_generator_env[i], gym.spaces.Dict):
                self.dict_space = True
                # TODO this doesn't work. See externalized memory for how to do this.
                low = dict(env_obs_spaces[i].low)
                low["her_obs"] = self.goal_generator_env.observation_space[i].low
                high = dict(env_obs_spaces[i].high)
                high["her_obs"] = self.goal_generator_env.observation_space[i].high
                dtype = dict(env_obs_spaces[i].dtype)
                dtype["her_obs"] = self.goal_generator_env.observation_space[i].dtype
                self.observation_space.append(gym.spaces.Dict(low=low, high=high, dtype=dtype))
                self.elusive_goal = []
                self.elusive_goal.append({k:np.zeros(v.shape, dtype=v.dtype) for k, v in self.goal_generator_env.observation_space[i].items()}) # The elusive goal can never actually be reached so that the q values given this goal are just the real q values from the game.
            else:
                self.observation_space.append(gym.spaces.Box(low=np.append(self.goal_generator_env.observation_space[i].low, env_obs_spaces[i].low), high=np.append(self.goal_generator_env.observation_space[i].high, env_obs_spaces[i].high), dtype=env_obs_spaces[i].dtype))
                self.elusive_goal = [np.zeros(self.goal_generator_env.observation_space[i].shape, dtype=self.goal_generator_env.observation_space[i].dtype) for i in range(self.n_agents)]

        self.her_reward = her_reward
        self.k = k
        self.episode_transitions = []
        self.her_transitions = []
        self.split_reward_streams = split_reward_streams

    def reset(self, **kwargs):
        self.episode_transitions = []
        self.her_transitions = []
        obs = self.env.reset(**kwargs)
        self.orig_s_prime = obs
        combined_obs = self.add_her_goal_to_state(self.elusive_goal, obs)
        return combined_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)        
        
        self.orig_s = self.orig_s_prime
        self.orig_s_prime = obs
        self.action = action
        self.rew = rew
        self.done = done

        if not self.test_mode:
            self.episode_transitions.append((self.orig_s, self.action, self.rew, self.orig_s_prime, self.done, self.goal_generator_env.get_goal()))      
            if done:
                self.her_transitions = self.__generate_transitions()
        
        combined_obs = self.add_her_goal_to_state(self.elusive_goal, obs)

        return combined_obs, rew, done, info

    def __generate_transitions(self):
        her_transitions = []
        for i in range(self.k):
            rand_trans_idx = np.random.randint(len(self.episode_transitions))
            s_g, a_g, r_g, s_prime_g, done_g, goal_g = self.episode_transitions[rand_trans_idx]
            for j in range(len(self.episode_transitions)):
                if j == rand_trans_idx:
                    s_g_her = self.add_her_goal_to_state(goal_g, s_g)
                    s_prime_g_her = self.add_her_goal_to_state(goal_g, s_prime_g)
                    if self.split_reward_streams:
                        her_transitions.append((s_g_her, a_g, r_g + (self.her_reward, 0), s_prime_g_her, done_g))
                    else:
                        her_transitions.append((s_g_her, a_g, r_g + self.her_reward, s_prime_g_her, done_g))
                else:
                    s, a, r, s_prime, done, goal_g = self.episode_transitions[j]
                    s = self.add_her_goal_to_state(goal_g, s)
                    s_prime = self.add_her_goal_to_state(goal_g, s_prime)
                    her_transitions.append((s, a, r, s_prime, done))
        return her_transitions

    def add_her_goal_to_state(self, goal, obs):
        combined_obs = []
        for i in range(self.n_agents):
            if self.dict_space:
                obs_dict = dict(obs[i])
                obs_dict["her_obs"] = goal[i]
                combined_obs.append(obs_dict)
            else:
                combined_obs.append(np.append(goal[i], obs[i]))
        return combined_obs

    def get_her_transitions(self, agent_nbr):
        return [(s[agent_nbr], a[agent_nbr], r[agent_nbr], s_prime[agent_nbr], done[agent_nbr]) for s, a, r, s_prime, done in self.her_transitions]