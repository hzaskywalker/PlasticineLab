import argparse
import os

import torch
from torch.utils.data.dataloader import DataLoader

from ...neurals.autoencoder import PCNEncoder
from ...neurals.latent_forward import ForwardModel
from ...neurals.pcdataloader import ChopsticksCFMDataset
from .cpc_loss import InfoNCELoss

device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--num_iters',type=int, default=10)
args = parser.parse_args()

dataset = ChopsticksCFMDataset()
dataloader = DataLoader(dataset,batch_size = args.batch_size)

n_particles = dataset.n_particles
n_actions = dataset.n_actions
latent_dim = 1024

encoder = PCNEncoder(
    state_dim=3,
    latent_dim=latent_dim).to(device)

forward_model = ForwardModel(
    latent_dim=latent_dim,
    # complexityLevel=ForwardModel.ComplexityLevel.Linear,
    action_dim=n_actions).to(device)

loss_fn = InfoNCELoss()

params = list(encoder.parameters()) + list(forward_model.parameters())
optimizer = torch.optim.Adam(params,lr=0.0001)

def train(encoder:PCNEncoder,
        forward_model:torch.nn.Module,
        optimizer: torch.nn.Module,
        dataloader:DataLoader,
        loss_fn: torch.nn.Module):

    total_loss = 0
    batch_cnt = 0
    for state, target, action in dataloader:
        state = state.to(device)
        target = target.to(device)
        action = action.to(device)
        optimizer.zero_grad()
        latent = encoder(state)
        print(latent.shape)
        print(action.shape)
        latent_pred = forward_model(latent, action)
        latent_next = encoder(target)
        loss = loss_fn(latent, latent_pred, latent_next)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
        batch_cnt += 1
    return total_loss/batch_cnt


for iter in range(args.num_iters):
    loss = train(encoder,forward_model,optimizer,dataloader, loss_fn)
    print("Iteration:",iter,"Loss:",loss)
if not os.path.exists('pretrain_model'):
    os.makedirs('pretrain_model')
torch.save(encoder.state_dict(),'pretrain_model/weight_cfm.pth')
