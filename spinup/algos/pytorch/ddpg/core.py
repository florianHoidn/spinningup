import numpy as np
import gym

import torch
import torch.nn as nn

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
        self.act_limit = torch.nn.Parameter(torch.from_numpy(np.abs(act_limit[0])), requires_grad=False)

    def forward(self, obs_in):
        if isinstance(obs_in, dict):
            obs = torch.cat(list(observation_space_in.values()), dim=-1)
        else:
            obs = obs_in
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, model, model_kwargs):
        super().__init__()
        model_kwargs["input_sizes"] = [obs_dim + act_dim] + model_kwargs["input_sizes"]
        model_kwargs["output_sizes"] += [1]
        self.q = model(**model_kwargs)

    def forward(self, obs_in, act):
        if isinstance(obs_in, dict):
            obs = torch.cat(list(observation_space_in.values()), dim=-1)
        else:
            obs = obs_in
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ActorCritic(nn.Module):

    def __init__(self, observation_space_in, action_space, model, model_kwargs_getter):
        super().__init__()

        #TODO I need to handle dictionary spaces properly
        if isinstance(observation_space_in, gym.spaces.Dict):
            #TODO handle low and high boundaries properly.
            observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.concatenate([o.shape for o in observation_space_in.spaces.values()]))
        else:
            observation_space = observation_space_in

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        #act_limit = action_space.high[0]
        act_limit = (action_space.low, action_space.high)

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, model, model_kwargs_getter(), act_limit)
        self.q = QFunction(obs_dim, act_dim, model, model_kwargs_getter())

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
