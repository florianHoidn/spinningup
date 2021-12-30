import gym
import numpy as np

class SignalingEnv(gym.Wrapper):
    """
    This environment augments the action vector and the observation space, in a way that allows
    the agents to store latent information in the action vector and receive it as an additional 
    obervational input in the next step. The hope is that this kind of recurrency, combined with
    RL bootstrapping, can replace actual recurrent blocks like LSTMs and serve as a 
    signaling channel in the agents' models.
    """
    def __init__(self, env, signal_vector_size, signal_stack_size=1):
        super().__init__(env)
        self.signal_vector_size = signal_vector_size
        self.signal_stack_size = signal_stack_size

        env_obs_spaces = env.observation_space
        env_act_spaces = env.action_space
        if not type(env_obs_spaces) is list:
            env_obs_spaces = [env_obs_spaces]
            env_act_spaces = [env_act_spaces]
            self.multi_agent_parent = True
        else:
            self.multi_agent_parent = False

        # Step 1: augment the observation space:
        self.dict_space = False
        self.n_agents = len(env_obs_spaces)
        self.observation_space = []
        self.null_signal = []
        for i in range(self.n_agents):
            if isinstance(env_obs_spaces[i], gym.spaces.Dict):
                self.dict_space = True
                augmented_obs_space_dict = dict(env_obs_spaces[i].spaces)
                low = -np.ones(self.signal_vector_size * self.signal_stack_size)
                high = np.ones(self.signal_vector_size * self.signal_stack_size)
                augmented_obs_space_dict["signal"] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                self.observation_space.append(gym.spaces.Dict(augmented_obs_space_dict))
            else:
                self.observation_space.append(gym.spaces.Box(
                    low=np.concatenate([env_obs_spaces[i].low, -np.ones(self.signal_vector_size * self.signal_stack_size)]), 
                    high=np.concatenate([env_obs_spaces[i].high, np.ones(self.signal_vector_size * self.signal_stack_size)]), 
                    dtype=env_obs_spaces[i].dtype))
                
            self.null_signal.append(np.zeros(shape=(self.signal_vector_size,), dtype=env_act_spaces[i].dtype))
        self.signal_stack = {i:[self.null_signal[i] for _ in range(self.signal_stack_size)] for i in range(self.n_agents)}
        
        # step 2: augment the action space:
        self.signal_from_to_idx = []
        action_spaces = []
        for space in env_act_spaces:
            low = np.concatenate([-2 * np.ones(self.signal_vector_size), space.low * np.ones(space.shape) if np.isscalar(space.low) else space.low])
            high = np.concatenate([2 * np.ones(self.signal_vector_size), space.high * np.ones(space.shape) if np.isscalar(space.high) else space.high])
            action_spaces.append(gym.spaces.Box(low=low, high=high, dtype=space.dtype))
            self.signal_from_to_idx.append((space.shape[-1], space.shape[-1] + self.signal_vector_size))
        self.action_space = action_spaces

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.signal_stack = {i:[self.null_signal[i] for _ in range(self.signal_stack_size)] for i in range(self.n_agents)}
        if self.dict_space:
            obs_per_agent = []
            for i in range(self.n_agents):
                combined_obs = dict(obs[i])
                combined_obs["signal"] = np.concatenate(self.signal_stack[i], axis=-1)
                obs_per_agent.append(combined_obs)
            return obs_per_agent
        else:
            obs_per_agent = []
            for i in range(self.n_agents):
                obs_per_agent.append(np.concatenate((obs[i], np.concatenate(self.signal_stack[i], axis=-1)), axis=-1))
            return obs_per_agent

    def step(self, action):
        if not type(self.action_space) is list:
            action = np.array([action])
        obs, rew, done, info = self.env.step(action[:,self.signal_vector_size:])
        obs_per_agent = []
        if self.dict_space:
            for i in range(self.n_agents):
                combined_obs = dict(obs[i])
                signal = np.tanh(action[i][:self.signal_vector_size])
                #signal_tanh = np.tanh(action[i][:self.signal_vector_size])
                #prev_signal = self.signal_stack[i][-1]
                #signal = np.array([prev_signal[k] if signal_tanh[k] < 0 else signal_tanh[k] for k in range(self.signal_vector_size)])
                self.signal_stack[i].pop(0)
                self.signal_stack[i].append(signal)
                combined_obs["signal"] = np.concatenate(self.signal_stack[i], axis=-1)
                obs_per_agent.append(combined_obs)
        else:
            obs_per_agent = []
            for i in range(self.n_agents):
                signal = np.tanh(action[i][:self.signal_vector_size])
                #signal_tanh = np.tanh(action[i][:self.signal_vector_size])
                #prev_signal = self.signal_stack[i][-1]
                #signal = np.array([prev_signal[k] if signal_tanh[k] < 0 else signal_tanh[k] for k in range(self.signal_vector_size)])
                self.signal_stack[i].pop(0)
                self.signal_stack[i].append(signal)
                obs_per_agent.append(np.concatenate((obs[i], np.concatenate(self.signal_stack[i], axis=-1)), axis=-1))

        #print(f"sending: {signal}")

        return obs_per_agent, rew, done, info

    def get_signal_from_to_idx(self, agent_nbr):
        return self.signal_from_to_idx[agent_nbr]
