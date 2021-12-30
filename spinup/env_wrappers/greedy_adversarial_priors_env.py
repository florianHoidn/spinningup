import gym
import numpy as np
import os

class GreedyAdversarialPriorsEnv(gym.Wrapper):
    """
    An environment that provides "expert" demonstrations for training a discriminator,
    - just like in the AmpEnv for adversarial motion prior.
    The idea is, though, to always use the best k trajectories of the agent itself as "expert" trajectories.
    This should force the agents to completely focus on reproducing these comparatively successful trajectories
    and then explore from there.
    """
    def __init__(self, env, top_k=5, max_sequence_length=20, block_env_returns=False):
        super().__init__(env)

        self.multi_agent_parent = type(env.observation_space) is list
        self.observation_space = env.observation_space if self.multi_agent_parent else [env.observation_space]
        self.action_space = env.action_space if self.multi_agent_parent else [env.action_space]
        self.multi_modal_parent = type(self.observation_space[0]) 
        self.n_agents = len(self.observation_space)

        obs_space = self.observation_space[0]
        if type(obs_space) is gym.spaces.Dict:
            self.amp_obs_size = {} #{k:np.array(v.shape) for k,v in obs_space.spaces.items()}
            self.amp_obs_size.update({"action":np.array(self.action_space[0].shape)})
            self.amp_obs_size.update({("prev_" + k):np.array(v.shape) for k,v in obs_space.spaces.items()})
            self.multi_modal_parent = True
        else:
            #self.amp_obs_size = {"obs":np.array(obs_space.shape)}
            self.amp_obs_size = {"prev_obs": np.array(obs_space.shape), "action":np.array(self.action_space[0].shape)}
            #self.amp_obs_size = {"prev_obs": np.array(obs_space.shape), "action":np.array(self.action_space[0].shape), "obs":np.array(obs_space.shape)}
            self.multi_modal_parent = False

        self.top_k = top_k
        #self.discounted_fraction = int(np.floor(self.top_k * 0.2))
        self.max_sequence_length = max_sequence_length
        self.zero_rews = [0] * self.n_agents if block_env_returns else None
        self.current_min_k_idx = None

        self.top_sequences = {i:[] for i in range(self.n_agents)}
        self.top_actions = {i:[] for i in range(self.n_agents)}
        self.top_returns = {i:[] for i in range(self.n_agents)}
        self.current_sequence = {i:[] for i in range(self.n_agents)}
        self.current_actions = {i:[] for i in range(self.n_agents)}
        #self.current_return = {i:0 for i in range(self.n_agents)}
        self.current_rewards = {i:[] for i in range(self.n_agents)}
        self.hidden_step_rewards = [0] * self.n_agents
        self.prev_state_amp_agent = {i:None for i in range(self.n_agents)}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_sequence = {i:[obs[i] if self.multi_agent_parent else obs] for i in range(self.n_agents)}
        self.current_actions = {i:[] for i in range(self.n_agents)}
        #self.current_return = {i:0 for i in range(self.n_agents)}
        self.current_rewards = {i:[] for i in range(self.n_agents)}
        self.current_min_k_idx = None
        return obs

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions if self.multi_agent_parent else actions[0])
        if not self.multi_agent_parent:
            obs, rew, done, info = [obs], [rew], [done], {'n':[info]}
        self.hidden_step_rewards = rew
        for i in range(self.n_agents):
            self.current_sequence[i].append(obs[i])
            self.current_actions[i].append(actions[i])
            #self.current_return[i] += rew[i]
            self.current_rewards[i].append(rew[i])
            self.store_prev_state_amp_agent(i)
            if done[i] or len(self.current_sequence[i]) > self.max_sequence_length:
                curr_return = np.sum(self.current_rewards[i])
                if len(self.top_returns[i]) < self.top_k:
                    #self.top_returns[i].append(self.current_return[i])
                    self.top_returns[i].append(curr_return)
                    self.top_sequences[i].append(np.array(self.current_sequence[i]))
                    self.top_actions[i].append(np.array(self.current_actions[i]))
                else:
                    
                    # Let's discount some of the top returns a bit so that we cycle through sequences frequently.
                    #min_k_idx = np.argpartition(self.top_returns[i], self.discounted_fraction)[:self.discounted_fraction]
                    discount = np.abs(np.max(self.top_returns[i])) * 0.00001
                    self.top_returns[i] = [ret - discount for ret in self.top_returns[i]] # Makes sure that the returns always decrease over time.
                    # Or simpler, just discount everythin by a constant factor:
                    # TODO careful, this doesn't work for negative returns.
                    #self.top_returns[i] = [ret * 0.995 for ret in self.top_returns[i]]

                    min_k_idx = np.argmin(self.top_returns[i]) if self.current_min_k_idx is None else self.current_min_k_idx
                    if self.top_returns[i][min_k_idx] <= curr_return: #self.current_rewards[i]:
                        self.top_returns[i][min_k_idx] = curr_return #self.current_return[i]
                        self.top_sequences[i][min_k_idx] = np.array(self.current_sequence[i])
                        self.top_actions[i][min_k_idx] = np.array(self.current_actions[i])
                        self.current_min_k_idx = min_k_idx # We don't want to spam the entire k entries with basically the same trajectory, so we stick with it within an episode.
                if not done[i]: # otherwise reset() will do this anyway.
                    self.resize_sequences(i, obs)
        return obs, rew if self.zero_rews is None else self.zero_rews, done, info

    def resize_sequences(self, i, obs):
        #self.current_sequence[i] = [obs[i] if self.multi_agent_parent else obs]
        #self.current_actions[i] = []
        #self.current_return[i] = 0

        self.current_sequence[i].pop(0)
        self.current_actions[i].pop(0)
        self.current_rewards[i].pop(0)
        
    def get_amp_obs(self, i):
        if len(self.current_actions[i]) == 0:
            return None
        if self.multi_modal_parent:
            state_agent = {} #dict(self.current_sequence[i][-1])
            state_agent.update({"action":self.current_actions[i][-1]})
            state_agent.update({("prev_" + k):v for k,v in self.current_sequence[i][-2].items()})

            expert_state_available = False
            if len(self.top_sequences[i]) > 0:
                seq_idx = np.random.randint(0, len(self.top_sequences[i]))
                seq = self.top_sequences[i][seq_idx]
                actions = self.top_actions[i][seq_idx]
                if len(seq) >= 1 and len(actions) >= 1: # Only needed when we use transitions instead of just the current state.
                    expert_state_available = True
                    step_idx = np.random.randint(1, len(seq)) #np.random.randint(0, len(seq))
                    prev_obs = {("prev_" + k):v for k,v in seq[step_idx - 1].items()}
                    obs = seq[step_idx]
                    state_expert = {} #dict(obs)
                    state_expert.update({"action":actions[step_idx - 1]})
                    state_expert.update(prev_obs)
            
            if not expert_state_available:
                state_expert = state_agent
        else:
            state_agent = {} #{"obs":self.current_sequence[i][-1]}
            state_agent.update({"action":self.current_actions[i][-1]})
            state_agent.update({"prev_obs":self.current_sequence[i][-2]})
            
            expert_state_available = False
            if len(self.top_sequences[i]) > 0:
                seq_idx = np.random.randint(0, len(self.top_sequences[i]))
                seq = self.top_sequences[i][seq_idx]
                actions = self.top_actions[i][seq_idx]
                if len(seq) >= 1 and len(actions) >= 1: # Same as above
                    expert_state_available = True
                    step_idx = np.random.randint(1, len(seq)) #np.random.randint(0, len(seq))
                    prev_obs = {"prev_obs":seq[step_idx - 1]}
                    obs = seq[step_idx]
                    state_expert = {} #{"obs":obs}
                    state_expert.update({"action":actions[step_idx - 1]})
                    state_expert.update(prev_obs)
            
            if not expert_state_available:
                state_expert = state_agent

        amp_obs_stack = []
        if hasattr(self.env, "get_amp_obs"):
            amp_obs_stack = self.env.get_amp_obs(i)
        return amp_obs_stack + [{"state_amp_agent":state_agent, "state_amp_expert":state_expert}]

    def store_prev_state_amp_agent(self, i):
        if self.multi_modal_parent:
            state_agent = {} #{k:[v] for k,v in self.current_sequence[i][-1].items()}
            state_agent.update({"action":[self.current_actions[i][-1]]})
            state_agent.update({("prev_" + k):[v] for k,v in self.current_sequence[i][-2].items()})
        else:
            state_agent = {} #{"obs":[self.current_sequence[i][-1]]}
            state_agent.update({"action":[self.current_actions[i][-1]]})
            state_agent.update({"prev_obs":[self.current_sequence[i][-2]]})
        self.prev_state_amp_agent[i] = state_agent
        
    def get_prev_state_amp_agent(self, i):
        return self.prev_state_amp_agent[i]

    def get_amp_obs_size(self):
        return self.amp_obs_size

    def stack_amp_replay_buffers(self, replay_buffer, model, imitation_reward_weight, restore_model_paths, agent_nbr, logger, cuda_device):
        if hasattr(self.env, "stack_amp_replay_buffers"):
            replay_buffer = self.env.stack_amp_replay_buffers(replay_buffer, model, imitation_reward_weight, restore_model_paths[:-1] if restore_model_paths is not None else None, agent_nbr, logger, cuda_device)

        from spinup.replay_buffers.amp_replay_buffer_pytorch import AMPReplayBuffer
        return AMPReplayBuffer(inner_replay_buffer=replay_buffer, 
            discriminator_model_fn=model.net, discriminator_model_kwargs_getter=model.get_tiny_kwargs,
            restore_model_path=restore_model_paths[-1] if restore_model_paths is not None and len(restore_model_paths) >= 1 else None,
            save_model_path=os.path.join(logger.output_dir, "pyt_save", "amp_discriminator_model_" + str(agent_nbr) + "_" + self.__class__.__name__ + ".pt"),
            imitation_reward_weight=imitation_reward_weight, amp_env=self, logger=logger, cuda_device=cuda_device)

    #def get_hidden_step_rewards(self):
    #    return np.array(self.hidden_step_rewards)