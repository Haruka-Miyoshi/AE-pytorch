import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder

class Model(nn.Module):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encode = Encoder(self.x_dim, self.h_dim, self.z_dim)
        self.decode = Decoder(self.x_dim, self.h_dim, self.z_dim)
    
    def forward(self, x):
        z = self.encode(x)
        xh = self.decode(z)
        return z, xh