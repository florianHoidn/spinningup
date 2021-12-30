import torch.nn as nn
import numpy as np

def net(input_sizes, hidden_sizes, output_sizes, activation, output_activation, linear_layer=nn.Linear, dropout_rate=0):
    input_layer_size = input_sizes[0] if np.isscalar(input_sizes[0]) else input_sizes[0][0]
    sizes = [input_layer_size] + hidden_sizes + output_sizes
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if dropout_rate > 0:
            layers += [linear_layer(sizes[j], sizes[j+1]), act(), nn.Dropout(dropout_rate)]
        else:
            layers += [linear_layer(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def get_default_kwargs():
    return {"input_sizes":[], "hidden_sizes":[512,512,512,512], "output_sizes":[], "activation":nn.ReLU, "output_activation":nn.Identity}

def get_tanh_kwargs():
    return {"input_sizes":[], "hidden_sizes":[1024,1024,1024], "output_sizes":[], "activation":nn.Tanh, "output_activation":nn.Identity}

def get_tiny_kwargs():
    return {"input_sizes":[], "hidden_sizes":[128,128], "output_sizes":[], "activation":nn.ReLU, "output_activation":nn.Identity}