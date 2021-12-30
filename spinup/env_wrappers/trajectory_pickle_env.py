import gym
import pickle
import os
import numpy as np

class TrajectoryPickleEnv(gym.Wrapper):
    """
    Records all transitions and pickles them so that they can be used for 
    to train a decision transformer offline. 
    """
    def __init__(self, env, output_pickle_trajectories, save_freq=10000, max_nbr_transitions=5e6, offline_training_keys = ["observations", "next_observations", "actions", "rewards", "terminals"]):
        super().__init__(env)
        self.output_pickle_trajectories = output_pickle_trajectories
        self.save_freq = save_freq
        self.prev_obs = None
        self.max_nbr_transitions = max_nbr_transitions
        self.offline_training_keys = offline_training_keys
        pickle_dir = os.path.dirname(self.output_pickle_trajectories)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        self.step_counter = 0 
        self.n_agents = len(self.env.observation_space) if type(self.env.observation_space) is list else None
        self.trajectories = []
        self.traj_data = {k:[] for k in offline_training_keys}
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions)
        if self.n_agents is None:
            self.record_transition(self.prev_obs, obs, actions, rew, done)
        else:
            for i in range(self.n_agents):
                self.record_transition(self.prev_obs[i], obs[i], actions[i], rew[i], done[i])
        self.prev_obs = obs
        return obs, rew, done, info

    def record_transition(self, obs, next_obs, act, rew, done):
        if len(self.trajectories) > self.max_nbr_transitions:
            self.trajectories.pop(0)

        #if type(obs) is dict:
        #    self.traj_data["observations"].append(obs["obs"])
        #    self.traj_data["next_observations"].append(next_obs["obs"])
        #else:
        self.traj_data["observations"].append(obs)
        self.traj_data["next_observations"].append(next_obs)
        self.traj_data["actions"].append(act)
        self.traj_data["rewards"].append(rew)
        self.traj_data["terminals"].append(done)
        if done:
            self.trajectories.append({k:np.array(v) for k,v in self.traj_data.items()})
            self.traj_data = {k:[] for k in self.offline_training_keys}

        self.step_counter += 1
        if self.step_counter % self.save_freq == 0:
            with open(self.output_pickle_trajectories, "wb") as f:
                pickle.dump(self.trajectories, f)
