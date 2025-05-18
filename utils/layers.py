import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        stride = output_dim // input_dim
        stride_list = [2] * (int(math.log2(stride)) - 1)
        self.layers = nn.ModuleList()
        for s in stride_list:
            self.layers.append(
                nn.Sequential(nn.Conv2d(input_dim, input_dim * s, kernel_size=1), nn.BatchNorm2d(input_dim * s),
                              nn.ReLU()))
            input_dim *= s
        self.layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x