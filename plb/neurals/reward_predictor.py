import torch
import torch.nn as nn
import torch.nn.functional as F

# A Simple Three Layer MPI
# Since the loss is compute within taichi, unable to make minibatch training
class RPMLP(nn.Module):
    def __init__(self,latent_dim):
        super(RPMLP,self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.latent_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self,z):
        h = self.fc1(z)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        out = self.fc3(h)
        return out
        