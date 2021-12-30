import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spinup.models.pytorch.resnet import ResNet

class NoisyLinear(nn.Linear):
    """
    A linear layer with trainable Gaussian noise on the weights.
    See https://arxiv.org/abs/1706.10295 on why this type of parameter space noise
    can be a very effective way to improve exploration in RL agents. 
    """
    def __init__(self, in_features, out_features, bias=True):
        self.sigma_weight, self.sigma_bias = None, None
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))#, requires_grad=False)
        self.register_parameter('sigma_weight', self.sigma_weight) # TODO I thought that this shouldn't really be needed, but let's test.
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features))#, requires_grad=False)
            self.register_parameter('sigma_bias', self.sigma_bias)
        else:
            self.register_parameter('sigma_bias', None)
        self.init_noise_scale = 1#1e-3
        self.weight_noise = None
        self.bias_noise = None
        self.reset_noise_parameters()
        self.generate_parameter_noise(device=self.sigma_weight.device)
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.sigma_weight is not None:
            self.reset_noise_parameters()

    def reset_noise_parameters(self):
        # Let's initialize the sigmas just like the weights and biases themselves.
        fan_sigma = nn.init._calculate_correct_fan(self.sigma_weight, 'fan_in')
        gain_sigma = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std_sigma = gain_sigma / np.sqrt(fan_sigma)
        bound_sigma = self.init_noise_scale * np.sqrt(3.0) * std_sigma
        with torch.no_grad():
            nn.init.uniform_(self.sigma_weight, -bound_sigma, bound_sigma)
        if self.sigma_bias is not None:
            fan_bias, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma_weight)
            bound_bias = self.init_noise_scale / np.sqrt(fan_bias) if fan_bias > 0 else 0
            with torch.no_grad():
                nn.init.uniform_(self.sigma_bias, -bound_bias, bound_bias)

    def forward(self, input):
        if self.weight_noise is None:
            return super().forward(input)

        self.generate_parameter_noise(device=self.sigma_weight.device) # In the anyrl framework, noise was generated like this on every forward pass.
        
        noisy_weights = self.weight + self.sigma_weight * self.weight_noise
        if self.bias is None:
            return F.linear(input, noisy_weights, self.bias)
        noisy_bias = self.bias + self.sigma_bias * self.bias_noise
        return F.linear(input, noisy_weights, noisy_bias)

    def generate_parameter_noise(self, device):
        self.weight_noise = torch.randn(size=(self.out_features, self.in_features), device=device, requires_grad=False)
        if self.bias is not None:
            self.bias_noise = torch.randn(size=(self.out_features,), device=device, requires_grad=False)

# TODO rename to noisy_linear.py and treat independently from resnet.
#def net(input_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation):
#    return ResNet(input_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, linear_layer=NoisyLinear)

#def get_default_kwargs():
#    return {"input_sizes":[], "hidden_sizes":[256,256], "num_inner_res_blocks":3, "output_sizes":[], "activation":nn.ReLU, "output_activation":nn.Identity}
