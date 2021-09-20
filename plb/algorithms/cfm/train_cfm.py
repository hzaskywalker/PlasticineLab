import argparse
import os
from typing import Tuple, Union

import torch
from torch.utils.data.dataloader import DataLoader

from ...neurals.autoencoder import PCNEncoder
from ...neurals.latent_forward import ForwardModel, InverseModel
from ...neurals.pcdataloader import CFMDataset
from .cpc_loss import ForwardLoss, InfoNCELoss, InverseLoss, OriginInfoNCELoss

device = torch.device('cuda')

STATE_DIM  = 3
LATENT_DIM = 1024

def _complexity_2_enum(complexity: int) -> ForwardModel.ComplexityLevel:
    if complexity == 0: return ForwardModel.ComplexityLevel.MLP
    if complexity == 1: return ForwardModel.ComplexityLevel.SimpleMLP
    if complexity == 2: return ForwardModel.ComplexityLevel.Linear

    raise NotImplementedError("only 0, 1, 2 are supported")

def _complexity_and_loss_2_str(complexity: int, loss: str) -> str:
    if loss == 'OriginInfoNCELoss': return "origin_loss"
    if complexity == 0: return f"cfm_{loss}"
    if complexity == 1: return f"smaller_{loss}"
    if complexity == 2: return f"linear_{loss}"


def _preparation(datasetName: str, batchSize: int, complexity: int) -> Tuple[DataLoader, PCNEncoder, ForwardModel]:
    dataset = CFMDataset(datasetName) 
    dataloader = DataLoader(dataset, batch_size = batchSize)

    print(f"{dataset.n_particles} particles loaded from dataset:data/{datasetName}.npz")
    nActions = dataset.n_actions

    encoder = PCNEncoder(
        state_dim  = STATE_DIM,
        latent_dim = LATENT_DIM
    ).to(device)

    forwardModel = ForwardModel(
        latent_dim=LATENT_DIM,
        complexityLevel=_complexity_2_enum(complexity), 
        action_dim=nActions
    ).to(device)

    inverseModel = InverseModel(
        latent_dim=LATENT_DIM,
        action_dim=nActions).to(device)

    return dataloader, encoder, forwardModel, inverseModel

def train(encoder:PCNEncoder,
        forward_model:torch.nn.Module,
        optimizer: torch.nn.Module,
        dataloader:DataLoader,
        loss_type,
        forward_loss_fn: torch.nn.Module = None,
        inverse_model: torch.nn.Module = None,
        inverse_loss_fn: torch.nn.Module = None):

    total_loss = 0
    batch_cnt = 0
    for state, target, action in dataloader:
        state = state.to(device)
        target = target.to(device)
        action = action.to(device)
        optimizer.zero_grad()
        latent = encoder(state)
        latent_pred = forward_model(latent, action)
        latent_next = encoder(target)
        loss = 0
        if loss_type in ['E2C','InfoNCE','OriginInfoNCE','Forward']:
            loss += forward_loss_fn(latent, latent_pred, latent_next)
        if loss_type in ['E2C','Inverse']:
            action_pred = inverse_model(latent,latent_next)
            loss += inverse_loss_fn(action_pred,action)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
        batch_cnt += 1
    return total_loss/batch_cnt


if __name__ == '__main__':
    assert os.path.exists('pretrain_model/cfm'), \
        "There must be an pretrain_model/cfm directory to store the training result"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    '-d', type=str, default='Chopsticks', help="dataset for CFM training, relative path to the `data` folder; NO .npz suffix needed")
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--num_iters',  '-n', type=int, default=10)
    parser.add_argument('--loss',       '-l', type=str, default="", help="<InfoNCE/OriginInfoNCE/Forward/Inverse/E2C>")
    parser.add_argument('--complexity', '-c', type=int, default=0, help="0 --- MLP; 1 --- SmallerMLP; 2 --- Linear")
    args = parser.parse_args()

    dataloader, encoder, forward_model, inverse_model = _preparation(args.dataset, args.batch_size, args.complexity)
    loss_fn_table = {'Forward':ForwardLoss,
                     'E2C':ForwardLoss,
                     'OriginInfoNCE':OriginInfoNCELoss,
                     'InfoNCE':InfoNCELoss}
    if args.loss != 'Inverse':
        forward_loss_fn = loss_fn_table[args.loss]()
    inverse_loss_fn = InverseLoss()
    


    params = list(encoder.parameters()) + list(forward_model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001)

    try: 
        for iter in range(args.num_iters):
            loss = train(encoder,forward_model,optimizer,dataloader, args.loss,forward_loss_fn=forward_loss_fn,inverse_model=inverse_model,inverse_loss_fn=inverse_loss_fn)
            print(f"Iteration:{iter}, Loss:{loss}")
    finally:
        pthName = f'pretrain_model/cfm/{args.dataset.lower()}_{_complexity_and_loss_2_str(args.complexity, args.loss)}/encoder'
        while os.path.exists(pthName + ".pth"):
            print(f"{pthName}.pth exists; saving as {pthName}_new.pth")
            pthName += "_new"
        torch.save(encoder.state_dict(), pthName + ".pth")
