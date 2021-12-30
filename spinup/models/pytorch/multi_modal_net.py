import torch
import torch.nn as nn
import numpy as np

from spinup.models.pytorch.resnet import ResNet

class MultiModalNet(nn.Module):
    """
    A model that handles multiple different input modalities. 
    The different inputs are specified by the first entry in input_sizes that should be of type dict. 
    Typical image inputs (shaped (H,W,C) or (H,W)) will be fed through a convolution, other data through a 
    fully connected layer before everything gets concatenated and fed through a ResNet.
    """
    def __init__(self, input_sizes, conv_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, noisy_linear_layers, dropout_rate):
        super().__init__()
        if isinstance(input_sizes[0], dict):
            sizes_dict = input_sizes[0]
            self.dict_space = True
        else:
            sizes_dict = {"obs":np.array(input_sizes[0])}
            self.dict_space = False

        self.permute_required = set()
        self.expand_dim_required = set()
        input_dict = {}
        self.modalities = []
        for modality, size in sizes_dict.items():
            self.modalities.append(modality)
            if type(size) == list:
                size = np.array(size)

            if np.ndim(size) == 0:
                input_dict[modality] = nn.Linear(size, input_sizes[1])
            elif size.shape[0] in [2, 3]:
                # let's treat 2d and 3d input tensors as images.
                if size.shape[0] == 3:
                    if size[2] in [1,3]:
                        in_channels = size[2]
                        self.permute_required.add(modality)
                    else:
                        # Otherwise, the color channel hopefully already is in the first position, as pytorch requires it.
                        in_channels = size[0]
                else:
                    in_channels = 1
                    self.expand_dim_required.add(modality)
                specialized_sub_net = []
                conv_out_size = np.array(size[:-1]) if modality in self.permute_required else np.array(size[1:]) # See section "Shape" at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html 
                for nbr_filters, kernel_size, stride, padding, dilation in conv_sizes:
                    specialized_sub_net.append(nn.Conv2d(in_channels=in_channels, out_channels=nbr_filters, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
                    specialized_sub_net.append(activation())
                    conv_out_size = np.floor((conv_out_size + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1)
                    in_channels = nbr_filters        
                specialized_sub_net.append(nn.Flatten())
                specialized_sub_net.append(nn.Linear(int(np.prod(conv_out_size) * in_channels), input_sizes[1]))
                input_dict[modality] = nn.Sequential(*specialized_sub_net)
            elif size.shape[0] == 1:
                input_dict[modality] = nn.Linear(size[0], input_sizes[1])
            else:
                print("MultiModalNet doesn't yet know how to handle input of size " + str(size) + ". Do the input images have the shape (H,W,C)?")
        self.input_subnets = nn.ModuleDict(input_dict)
        if noisy_linear_layers:
            from spinup.models.pytorch.noisy_resnet import NoisyLinear
            if num_inner_res_blocks > 0:
                self.hidden_layers = ResNet([len(sizes_dict) * input_sizes[1]], hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, linear_layer=NoisyLinear)
            else:
                import spinup.models.pytorch.mlp as mlp
                self.hidden_layers = mlp.net([len(sizes_dict) * input_sizes[1]], hidden_sizes, output_sizes, activation, output_activation, linear_layer=NoisyLinear, dropout_rate=dropout_rate)
        else:
            if num_inner_res_blocks > 0:
                self.hidden_layers = ResNet([len(sizes_dict) * input_sizes[1]], hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, dropout_rate=dropout_rate)
            else:
                import spinup.models.pytorch.mlp as mlp
                self.hidden_layers = mlp.net([len(sizes_dict) * input_sizes[1]], hidden_sizes, output_sizes, activation, output_activation, dropout_rate=dropout_rate)

    def forward(self, obs):
        multi_mod_obs = obs if self.dict_space else {"obs":obs}
        sub_activations = [self.input_subnets[modality](multi_mod_obs[modality].permute(0,3,1,2) if modality in self.permute_required 
                                                        else torch.unsqueeze(multi_mod_obs[modality], dim=1) if modality in self.expand_dim_required 
                                                        else multi_mod_obs[modality]) for modality in self.modalities]
        inner_representation = torch.cat(sub_activations, dim=-1)
        return self.hidden_layers(inner_representation)

def net(input_sizes, conv_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, noisy_linear_layers, dropout_rate):
    return MultiModalNet(input_sizes, conv_sizes, hidden_sizes, num_inner_res_blocks, output_sizes, activation, output_activation, noisy_linear_layers, dropout_rate)

def get_default_kwargs():
    return {"input_sizes":[1024], "conv_sizes":[(32, 8, 4, 0, 1), (64, 4, 2, 0, 1), (128, 2, 2, 0, 1)], "hidden_sizes":[1024, 1024], "num_inner_res_blocks":0, "output_sizes":[], "activation":nn.ReLU, "output_activation":nn.Identity, "noisy_linear_layers":False, "dropout_rate":0.05}
    
def get_tiny_kwargs():
    return {"input_sizes":[512], "conv_sizes":[(16, 8, 4, 0, 1), (32, 4, 2, 0, 1), (64, 2, 2, 0, 1)], "hidden_sizes":[512, 512], "num_inner_res_blocks":0, "output_sizes":[], "activation":nn.ReLU, "output_activation":nn.Identity, "noisy_linear_layers":False, "dropout_rate":0.01}

def get_tanh_kwargs():
    return {"input_sizes":[1024], "conv_sizes":[(32, 8, 4, 0, 1), (64, 4, 2, 0, 1), (128, 2, 2, 0, 1)], "hidden_sizes":[1024, 1024], "num_inner_res_blocks":0, "output_sizes":[], "activation":nn.Tanh, "output_activation":nn.Identity, "noisy_linear_layers":False, "dropout_rate":0}