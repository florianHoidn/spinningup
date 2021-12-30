import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete, Dict

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, model, model_kwargs):
        super().__init__()
        model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] += [act_dim]
        self.logits_net = model(**model_kwargs)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, model, model_kwargs):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] += [act_dim]
        self.mu_net = model(**model_kwargs)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):

    def __init__(self, obs_dim, model, model_kwargs):
        super().__init__()
        model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] += [1]
        self.v_net = model(**model_kwargs)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, model, model_kwargs_getter):
        super().__init__()


        if isinstance(observation_space, Dict):
            obs_dim = {k:np.array(v.shape) for k,v in observation_space.spaces.items()}
        else:
            obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = GaussianActor(obs_dim, action_space.shape[0], model, model_kwargs_getter())
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalActor(obs_dim, action_space.n, model, model_kwargs_getter())

        # build value function
        self.v  = Critic(obs_dim, model, model_kwargs_getter())

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]