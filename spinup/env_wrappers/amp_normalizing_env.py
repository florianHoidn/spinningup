import gym
import numpy as np
from spinup.env_wrappers.normalizing_env import NormalizingEnv, AvgTracker

#class AmpNormalizingEnv(NormalizingEnv):
class AmpNormalizingEnv(gym.Wrapper):
    """
    A NormalizingEnv that also normalizes amp observations.
    """
    def __init__(self, env, act_mean, act_var):
        #super().__init__(env, act_mean, act_var)
        super().__init__(env)
        
        self.n_agents = len(env.observation_space) if type(env.observation_space) is list else 1

        amp_obs_sizes = env.get_amp_obs_size()
        self.amp_dict_space = False
        if isinstance(amp_obs_sizes, dict):
            self.amp_dict_space = True
            self.amp_normalizers = []
            for _ in range(self.n_agents):
                self.amp_normalizers.append({obs_key:AvgTracker(size) for obs_key, size in amp_obs_sizes.items()})
        else:
            self.amp_normalizers = [AvgTracker(amp_obs_sizes) for _ in range(self.n_agents)]

    def get_amp_obs(self, i):
        amp_obs = self.env.get_amp_obs(i)
        if amp_obs is None:
            return None
        # TODO not sure if it would be better to normalize on a per batch level - DeepMimic does that, I think.
        agent_obs = amp_obs["state_amp_agent"]
        expert_obs = amp_obs["state_amp_expert"]
        if self.amp_dict_space:
            normalized_agent_obs = {}
            normalized_expert_obs = {}
            for obs_key, normalizer in self.amp_normalizers[i].items():
                normalizer.update_normalizer(agent_obs[obs_key])
                normalizer.update_normalizer(expert_obs[obs_key])
                normalized_agent_obs[obs_key] = normalizer.normalize(agent_obs[obs_key])
                normalized_expert_obs[obs_key] = normalizer.normalize(expert_obs[obs_key])
            return {"state_amp_agent": normalized_agent_obs, "state_amp_expert":normalized_expert_obs}
        else:
            self.amp_normalizers[i].update_normalizer(agent_obs)
            self.amp_normalizers[i].update_normalizer(expert_obs)
            return {"state_amp_agent": self.amp_normalizers[i].normalize(agent_obs), "state_amp_expert":self.amp_normalizers[i].normalize(expert_obs)}
