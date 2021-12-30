import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import MultiheadAttention, LayerNorm

import gym

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, model, model_kwargs, act_limit):
        super().__init__()
        model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] = [model_kwargs["hidden_sizes"][-1]]
        model_kwargs["output_activation"] = model_kwargs["activation"]
        self.net = model(**model_kwargs)
        self.mu_layer = nn.Linear(model_kwargs["output_sizes"][-1], act_dim)
        self.log_std_layer = nn.Linear(model_kwargs["output_sizes"][-1], act_dim)
        
        # TODO assumes the limits have mean 0 - but at least the limits per dim can be different (because act_limit=(low_vec, high_vec))
        self.act_limit = torch.nn.Parameter(torch.from_numpy(np.abs(act_limit[0])), requires_grad=False)
        self.act_dim = act_dim
        self.min_logp_pi = None
        self.max_logp_pi = None

    def forward(self, obs, deterministic=False, with_logprob=True, clamp_logprob=False, num_samples=1, sample_new_noise=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        
        # TODO let's try something a little bit more robust than the log stddev
        #log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #std = torch.exp(log_std)
        std = np.exp(LOG_STD_MIN) + (torch.sigmoid(log_std) * (np.exp(LOG_STD_MAX) - np.exp(LOG_STD_MIN)))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        # TODO let's try to get slightly more control over how deterministic we want the policy to be evaluated by sampling a couple of times.
        pi_action_sample_mean = None
        logp_pi_sample_mean = None

        for _ in range(num_samples):
            if deterministic:
                # Only used for evaluating policy at test time.
                pi_action = mu
            else:
                #TODO I feel like there might be a numerical issue with rsample, not sure though.
                #pi_action = pi_distribution.rsample()
                if with_logprob:
                    # During training, we sample from a fresh distribution every time.
                    pi_action = mu + std * Normal(torch.zeros(mu.size(), dtype=mu.dtype, device=mu.device), torch.ones(std.size(), dtype=std.dtype, device=std.device)).sample()
                else:
                    # At inference time, we stick with our sample for a while.
                    if sample_new_noise:
                        self.normal_samples = Normal(torch.zeros(mu.size(), dtype=mu.dtype, device=mu.device), torch.ones(std.size(), dtype=std.dtype, device=std.device)).sample()
                    pi_action = mu + std * self.normal_samples

            if with_logprob:
                # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
                # NOTE: The correction formula is a little bit magic. To get an understanding 
                # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
                # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
                # Try deriving it yourself as a (very difficult) exercise. :)
                logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

                logp_pi_sample_mean = logp_pi if logp_pi_sample_mean is None else logp_pi_sample_mean + logp_pi

            else:
                logp_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action

            pi_action_sample_mean = pi_action if pi_action_sample_mean is None else pi_action_sample_mean + pi_action

        pi_action_sample_mean /= num_samples

        if with_logprob:
            logp_pi_sample_mean /= num_samples
            
            if clamp_logprob:
                if self.max_logp_pi is None:
                    self.min_logp_pi = float(self.estimateMinLogpPi(dtype=mu.dtype, device=mu.device))
                    self.max_logp_pi = float(self.estimateMaxLogpPi(dtype=mu.dtype, device=mu.device))
                torch.clamp(logp_pi_sample_mean, self.min_logp_pi, self.max_logp_pi)
        #return pi_action, logp_pi
        return pi_action_sample_mean, logp_pi_sample_mean

    def estimateMaxLogpPi(self, dtype, device):
        return self.estimateLogpPi(stddev=LOG_STD_MIN, dtype=dtype, device=device)

    def estimateMinLogpPi(self, dtype, device):
        return self.estimateLogpPi(stddev=LOG_STD_MAX, dtype=dtype, device=device)

    def estimateLogpPi(self, stddev, dtype, device):
        with torch.no_grad():
            std_min = torch.exp(stddev * torch.ones(self.act_dim, dtype=dtype, device=device))
            pi_mean = torch.zeros(self.act_dim, dtype=dtype, device=device)
            # TODO mean zero doesn't account for tanh correction properly. So let's check the mean at action values.
            #pi_mean = torch.ones(self.act_dim, dtype=dtype, device=device) * 1e4
            almost_det_pi_distribution = Normal(pi_mean, std_min)
           
            #pi_samples = almost_det_pi_distribution.sample((1000,))
            pi_samples = torch.unsqueeze(pi_mean, dim=0)
           
            max_logp_pi = almost_det_pi_distribution.log_prob(pi_samples).sum(axis=-1)
            max_logp_pi -= (2*(np.log(2) - pi_samples - F.softplus(-2*pi_samples))).sum(axis=1)
            
            return torch.mean(max_logp_pi).cpu().numpy()

class QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, model, model_kwargs, split_reward_streams):
        super().__init__()
        if isinstance(obs_dim, dict):
            self.dict_space = True
            obs_with_action = dict(obs_dim)
            obs_with_action["action"] = np.array([act_dim])
            model_kwargs["input_sizes"] = [obs_with_action] + model_kwargs["input_sizes"]
        else:
            self.dict_space = False
            model_kwargs["input_sizes"] = [obs_dim + act_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] += [1 if not split_reward_streams else 2]

        # TODO checking if dropout helps the q function.
        #if "dropout_rate" in model_kwargs and model_kwargs["dropout_rate"] <= 0:
        #    model_kwargs["dropout_rate"] = 0.25

        self.q = model(**model_kwargs)

    def forward(self, obs, act):
        if self.dict_space:
            obs_input = dict(obs)
            obs_input["action"] = act
        else:
            obs_input = torch.cat([obs, act], dim=-1)
        q = self.q(obs_input)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, model, model_kwargs_getter, split_reward_streams=False):
        super().__init__()
        if isinstance(observation_space, gym.spaces.Dict):
            obs_dim = {k:np.array(v.shape) for k,v in observation_space.spaces.items()}
        else:
            obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        #act_limit = action_space.high[0]
        act_limit = (action_space.low, action_space.high)

        # build policy and value functions

        # TODO let's try no dropout on the policy
        
        pi_kwargs = model_kwargs_getter()
        pi_kwargs["dropout_rate"] = 0
        self.pi = SquashedGaussianActor(obs_dim, act_dim, model, pi_kwargs, act_limit)
        #self.pi = SquashedGaussianActor(obs_dim, act_dim, model, model_kwargs_getter(), act_limit)
        self.q1 = QFunction(obs_dim, act_dim, model, model_kwargs_getter(), split_reward_streams)
        self.q2 = QFunction(obs_dim, act_dim, model, model_kwargs_getter(), split_reward_streams)

    def act(self, obs, deterministic=False, sample_new_noise=True):
        with torch.no_grad():
            a, _ = self.pi(obs=obs, deterministic=deterministic, with_logprob=False, sample_new_noise=True)
            return a.cpu().numpy()

    def getMaxEntropyBonus(self, dtype, device):
        return self.pi.estimateMaxLogpPi(dtype, device)
