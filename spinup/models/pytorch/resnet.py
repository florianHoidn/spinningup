import torch
import torch.nn as nn
import numpy as np

class ResNet(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, linear_layer=nn.Linear, dropout_rate=0):
        super().__init__()
        """
        Implementation of a simple residual network.
        See https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32 for a tutorial on resnet architectures.
        """
        input_layer_size = input_sizes[0] if np.isscalar(input_sizes[0]) else input_sizes[0][0]
        self.input_layer = linear_layer(input_sizes[0], hidden_sizes[0])
        #self.input_layer = nn.Linear(input_sizes[0], hidden_sizes[0])
        res_blocks_list = []
        for i in range(num_inner_res_blocks):
            res_block = []
            for j in range(len(hidden_sizes)):
                prev_layer_size = hidden_sizes[j-1] if j > 0 else hidden_sizes[0]
                res_block += [activation(), linear_layer(prev_layer_size, hidden_sizes[j])]
                #res_block += [activation(), nn.Linear(prev_layer_size, hidden_sizes[j])]
            res_blocks_list += [nn.Sequential(*res_block)]
        self.res_blocks = nn.ModuleList(res_blocks_list)
        self.output_layer = nn.Sequential(*[activation(), linear_layer(hidden_sizes[-1], output_sizes[-1]), output_activation()])
        #self.output_layer = nn.Sequential(*[activation(), nn.Linear(hidden_sizes[-1], output_sizes[-1]), output_activation()])
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, obs):
        needs_squeeze = False
        if len(obs.size()) == 1:
            obs = torch.unsqueeze(obs, dim=0)
            needs_squeeze = True
        net_out = self.input_layer(obs)
        if self.dropout is not None:
            net_out = self.dropout(net_out)
        for j in range(len(self.res_blocks)):
            net_out = self.res_blocks[j](net_out) + net_out
            if self.dropout is not None:
                net_out = self.dropout(net_out)
        if needs_squeeze:
            return torch.squeeze(self.output_layer(net_out), dim=0)
        return self.output_layer(net_out)

def net(input_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation):
    return ResNet(input_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation)

def get_default_kwargs():
    return {"input_sizes":[], "hidden_sizes":[256,256], "num_inner_res_blocks":3, "output_sizes":[], "activation":nn.ReLU, "output_activation":nn.Identity}
