from enum import Enum

import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    class ComplexityLevel(Enum):
        MLP = 0
        SimpleMLP = 1
        Linear = 2

    def __init__(self, latent_dim:int, action_dim:int=0, complexityLevel:"ComplexityLevel"= ComplexityLevel.MLP):
        """ A MLP/Linera forward model for CFM architecture

        The model can be MLP with a hidden layer, without the hidden
        layer, or a single linear predictor, determined by complexityLevel

        :param latent_dim: the input dimension of the model, which, in
            CFM, is the shape of the learned latent space
        :param action_dim: the dimension of input action
        :param complexityLevel: a flag determining the model architecture
        """
        super().__init__()
        self.latent_dim = latent_dim
        _hidden_size = 64
        if complexityLevel == complexityLevel.MLP:
            self._model = nn.Sequential(
                nn.Linear(latent_dim + action_dim, _hidden_size),
                nn.ReLU(),
                nn.Linear(_hidden_size, _hidden_size),
                nn.ReLU(),
                nn.Linear(_hidden_size, latent_dim)
            )
        elif complexityLevel == complexityLevel.SimpleMLP:
            self._model = nn.Sequential(
                nn.Linear(latent_dim + action_dim, _hidden_size),
                nn.ReLU(),
                nn.Linear(_hidden_size, latent_dim)
            )
        else: 
            self._model = nn.Linear(latent_dim + action_dim, latent_dim)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self._model(x)
        return x

class InverseModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        x = torch.cat((z, z_next), dim=1)
        return self.model(x)