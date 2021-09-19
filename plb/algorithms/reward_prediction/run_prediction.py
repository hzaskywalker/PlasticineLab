import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from ...config import load
from yacs.config import CfgNode
from .utils import merge_lists
from ...neurals.reward_predictor import RPMLP
from ...neurals.autoencoder import PCNEncoder
from ...neurals.pcdataloader import ChopSticksDataset
from torch.utils.data.dataloader import DataLoader
from ...engine.losses import PredLoss

# Create loss instance, and initialize it
def load_cfg(load_path,version):
    assert version >= 1
    # Get current directory that the python file exist
    PATH = '/home/micrl/chensirui/OriginalCode/PlasticineLab/plb/envs/'
    cfg_path = os.path.join(PATH, load_path+'.yml')
    cfg = load(cfg_path)
    variants = cfg.VARIANTS[version - 1]

    new_cfg = CfgNode(new_allowed=True)
    new_cfg = new_cfg._load_cfg_from_yaml_str(yaml.safe_dump(variants))
    new_cfg.defrost()
    if 'PRIMITIVES' in new_cfg:
        new_cfg.PRIMITIVES = merge_lists(cfg.PRIMITIVES, new_cfg.PRIMITIVES)
    if 'SHAPES' in new_cfg:
        new_cfg.SHAPES = merge_lists(cfg.SHAPES, new_cfg.SHAPES)
    cfg.merge_from_other_cfg(new_cfg)

    cfg.defrost()
    # set target path id according to version
    name = list(cfg.ENV.loss.target_path)
    name[-5] = str(version)
    cfg.ENV.loss.target_path = os.path.join(PATH, '../', ''.join(name))
    cfg.VARIANTS = None
    cfg.freeze()

    return cfg

def state_preprocess(state,encoder,target_fn):
    state = state[0].squeeze()
    print("State Shape:",state.shape)
    target_loss = target_fn(state)
    latent = encoder(state.float())
    return latent,target_loss

# Need to define the training pipeline
# Only support batch size == 1 which is fair to everyone which is never mind
# The gradient will not pass through the encoder
def train(target_fn,loss_fn,encoder,predictor,dataloader,optimizer):
    total_loss = 0
    batch_cnt = 0
    for state, _, _, _ in dataloader:
        optimizer.zero_grad()
        latent,y_true = state_preprocess(state,encoder,target_fn)
        y_pred = predictor(latent)
        loss = loss_fn(y_true,y_pred)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        batch_cnt += 1
        #print("Batch:",batch_cnt)
    return total_loss/batch_cnt
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default = 'chopsticks')
    parser.add_argument('--version',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--epochs',type=int,default=5)
    args = parser.parse_args()
    
    cfg = load_cfg(args.config,args.version)
    target_fn = PredLoss(cfg)
    target_fn.initialize()

    encoder = PCNEncoder()
    for p in encoder.parameters():
        p.requires_grad_(False)
    predictor = RPMLP(1024)
    optimizer = torch.optim.Adam(predictor.parameters(),lr=args.lr)
    dataset = ChopSticksDataset()
    loss_fn = nn.MSELoss()
    dataloader = DataLoader(dataset,batch_size = 1)
    for epoch in range(args.epochs):
        loss = train(target_fn,loss_fn,encoder,predictor,dataloader,optimizer)
        print("Epoch: {} Loss: {}".format(epoch,loss))



    
    

    
