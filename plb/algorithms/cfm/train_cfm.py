import argparse
import os
from typing import Tuple, Union

import torch
import torch.nn
from torch.utils.data.dataloader import DataLoader

from ...neurals.autoencoder import PCNEncoder
from ...neurals.latent_forward import ForwardModel, InverseModel
from ...neurals.pcdataloader import CFMDataset
from .cpc_loss import ForwardLoss, InfoNCELoss, InverseLoss, OriginInfoNCELoss

device = torch.device('cuda')

STATE_DIM  = 3
LATENT_DIM = 1024
PRETRAIN_MODEL = "pretrain_model"
LOSS_FN_TABLE = {
    'Forward':ForwardLoss,
    'E2C':ForwardLoss,
    'OriginInfoNCE':OriginInfoNCELoss,
    'InfoNCE':InfoNCELoss
}
FORWARD_MODELS = {'E2C','InfoNCE','OriginInfoNCE','Forward'}
INVERSE_MODELS = {'E2C','Inverse'}

def _complexity_2_enum(complexity: int) -> ForwardModel.ComplexityLevel:
    if complexity == 0: return ForwardModel.ComplexityLevel.MLP
    if complexity == 1: return ForwardModel.ComplexityLevel.SimpleMLP
    if complexity == 2: return ForwardModel.ComplexityLevel.Linear

    raise NotImplementedError("only 0, 1, 2 are supported")

def _model_saving_path(lossType: str, dataset: str, complexity: Union[ForwardModel.ComplexityLevel, None] = None) -> str:
    lossTypeLower = lossType.lower()
    if lossTypeLower == 'inverse' or lossTypeLower == 'e2c' or lossTypeLower == 'forward':
        return os.path.join(PRETRAIN_MODEL, lossTypeLower, dataset, "encoder")
    elif lossTypeLower == 'origininfonce':
        return os.path.join(PRETRAIN_MODEL, "cfm", f"{dataset}_origin_loss", "encoder")
    else: # loss type is InfoNCELoss
        assert complexity is not None, \
            "For CFM encoder w/ InfoNCELoss, complexity MUST NOT BE NONE"
        if complexity == ForwardModel.ComplexityLevel.MLP:
            complexityStr = "cfm"
        elif complexity == ForwardModel.ComplexityLevel.SimpleMLP:
            complexityStr = "smaller"
        else:
            complexityStr = "linear"
        return os.path.join(PRETRAIN_MODEL, "cfm", f"{dataset}_{complexityStr}", encoder)


def _preparation(datasetName: str, batchSize: int, complexity: ForwardModel.ComplexityLevel) -> Tuple[DataLoader, PCNEncoder, ForwardModel]:
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
        complexityLevel=complexity, 
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
        if loss_type in FORWARD_MODELS:
            loss += forward_loss_fn(latent, latent_pred, latent_next)
        if loss_type in INVERSE_MODELS:
            action_pred = inverse_model(latent,latent_next)
            loss += inverse_loss_fn(action_pred,action)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
        batch_cnt += 1
    return total_loss/batch_cnt


if __name__ == '__main__':
    for modelClass in ['cfm', 'inverse', 'forward', 'e2c']:
        assert os.path.exists(os.path.join(PRETRAIN_MODEL, modelClass)), \
            f"There must be an pretrain_model/{modelClass} directory to store the training result"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    '-d', type=str, default='Chopsticks', help="dataset for CFM training, relative path to the `data` folder; NO .npz suffix needed")
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--num_iters',  '-n', type=int, default=10)
    parser.add_argument('--loss',       '-l', type=str, default="InfoNCE", help="[InfoNCE / OriginInfoNCE / Forward / Inverse / E2C]")
    parser.add_argument('--complexity', '-c', type=int, default=0, help="0 --- MLP; 1 --- SmallerMLP; 2 --- Linear")
    args = parser.parse_args()

    if args.complexity is not 0:
        print("\033[33mWARNING: the --complexity argument are only for CFM losses, i.e. either InfoNCE or OriginInfoNCE\033[0m")

    complexity = _complexity_2_enum(args.complexity)

    dataloader, encoder, forward_model, inverse_model = _preparation(args.dataset, args.batch_size, complexity)

    forward_loss_fn = None
    if args.loss != 'Inverse':
        forward_loss_fn = LOSS_FN_TABLE[args.loss]()
    inverse_loss_fn = InverseLoss()
    
    params = list(encoder.parameters()) + list(forward_model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001)

    try: 
        for iter in range(args.num_iters):
            loss = train(encoder,forward_model,optimizer,dataloader, args.loss,forward_loss_fn=forward_loss_fn,inverse_model=inverse_model,inverse_loss_fn=inverse_loss_fn)
            print(f"Iteration:{iter}, Loss:{loss}")
    finally:
        pthName = _model_saving_path(args.loss, args.dataset, complexity)
        while os.path.exists(pthName + ".pth"):
            print(f"{pthName}.pth exists; saving as {pthName}_new.pth")
            pthName += "_new"
        torch.save(encoder.state_dict(), pthName + ".pth")
