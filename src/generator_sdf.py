"""
Implementation of a simple DeepSDF.
"""

import torch
import torch.nn as nn

class SingleShapeDeepSDF(nn.Module):
    """
    DeepSDF network from Park et al. 2019.
    
    This version represents a single shape, and as such
    does not require a latent vector as input.
    """
    
    def __init__(self, hidden_dim=512, n_layers=8, in_insert=[4],
                 dropout=0.2, weight_norm=True, last_tanh=False):
        super().__init__()
        self.in_insert = in_insert

        self.fcs = nn.ModuleList()
        for n in range(n_layers):
            layer = []

            # Fully-connected
            in_d = out_d = hidden_dim
            if n == 0:
                in_d = 3
            if n == n_layers - 1:
                out_d = 1
            elif n + 1 in self.in_insert:
                out_d = hidden_dim - 3
            
            fc = nn.Linear(in_d, out_d)
            if weight_norm:
                fc = nn.utils.weight_norm(fc)
            layer.append(fc)

            # Activation
            if n < n_layers - 1:
                layer.append(nn.ReLU())
            elif last_tanh:
                layer.append(nn.Tanh())
            
            # Dropout
            if dropout > 0. and n < n_layers - 1:
                layer.append(nn.Dropout(dropout))

            # Combine them
            self.fcs.append(nn.Sequential(*layer))

    def forward(self, x):
        out = x

        for i, fc in enumerate(self.fcs):
            out = fc(out)

            if (i + 1) in self.in_insert:
               out = torch.cat([out, x], dim=-1)
        
        return out