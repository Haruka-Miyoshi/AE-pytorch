import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.x_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        xh = self.mlp(z)
        return xh