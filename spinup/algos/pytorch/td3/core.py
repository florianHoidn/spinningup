import numpy as np
import scipy.signal

import torch
import torch.nn as nn

import gym

#from spinup.models.pytorch.mlp import mlp

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, model, model_kwargs, act_limit):
        super().__init__()
        model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] += [act_dim]
        model_kwargs["output_activation"] = nn.Tanh
        self.pi = model(**model_kwargs)
        #self.act_limit = act_limit

        act_limit_size_halve = (act_limit[1] - act_limit[0]) * 0.5
        self.act_limit_mean = torch.nn.Parameter(torch.from_numpy(act_limit[0] + act_limit_size_halve), requires_grad=False)
        self.act_limit_scale = torch.nn.Parameter(torch.from_numpy(act_limit_size_halve), requires_grad=False)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit_mean + self.pi(obs) * self.act_limit_scale

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
        self.pi = Actor(obs_dim, act_dim, model, model_kwargs_getter(), act_limit)
        self.q1 = QFunction(obs_dim, act_dim, model, model_kwargs_getter(), split_reward_streams)
        self.q2 = QFunction(obs_dim, act_dim, model, model_kwargs_getter(), split_reward_streams)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
