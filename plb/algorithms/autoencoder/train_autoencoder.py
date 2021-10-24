import argparse
from functools import partial
from types import FunctionType
import torch
from torch.utils.data import DataLoader

from ...chamfer_distance import ChamferDistance
from ...engine.losses import compute_emd
from ...neurals import PointCloudAEDataset
from ...neurals import PCNAutoEncoder

device = torch.device("cuda:0")

def chamfer_loss(true_x,x_pred,chamfer_module):
    dist1,dist2 = chamfer_module(true_x,x_pred)
    loss = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    loss = loss.mean()*10
    return loss


def train(model,optimizer,loss_fn,dataloader):
    total_loss = 0
    batch_cnt = 0
    for x in dataloader:
        optimizer.zero_grad()
        x = x.float().to(device)
        x_hat = model(x)
        if args.loss == 'emd':
            loss,_ = loss_fn(x.permute(0,2,1),x_hat)
        else:
            loss = loss_fn(x.permute(0,2,1),x_hat)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        batch_cnt += 1
        #print(f"Batch:{batch_cnt}")
    return total_loss / batch_cnt


def main(
    loss: str,
    iters: int,
    savedModel: str,
    expName: str,
    dataset: str,
    freezeEncoder: bool, 
    loggerFunc: FunctionType = print
):

    assert(dataset in ['old_dataset/chopsticks','old_dataset/rope','torus-v1','writer-v1'])
    dataset = PointCloudAEDataset('data/{}.npz'.format(dataset))
    dataloader = DataLoader(dataset,batch_size=20)
    model = PCNAutoEncoder(dataset.n_particles,latent_dim=1024,hidden_dim=1024)
    if freezeEncoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    if savedModel != None:
        if savedModel.endswith('encoder'):
            model.encoder.load_state_dict(torch.load('pretrain_model/{}.pth'.format(savedModel)))
            print("Loaded Encoder!!")
        elif savedModel.endswith('decoder'):
            model.decoder.load_state_dict(torch.load('pretrain_model/{}.pth'.format(savedModel)))
            print("Loaded Decoder!!")
        else:
            model.load_state_dict(torch.load('pretrain_model/{}.pth'.format(savedModel)))
            print("Loaded Full model!!")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)

    if loss == 'chamfer':
        chamfer_module = ChamferDistance()
        loss_fn = partial(chamfer_loss,chamfer_module=chamfer_module)
    else:
        loss_fn = partial(compute_emd,iters=100)

    try:    
        for i in range(iters):
            epoch_loss = train(model,optimizer,loss_fn,dataloader)
            loggerFunc("Epoch {} Loss: {}".format(i,epoch_loss))
    except KeyboardInterrupt:
        print("Training is interrupted!")
    finally:
        torch.save(model.state_dict(),'pretrain_model/{}_whole.pth'.format(expName))
        torch.save(model.encoder.state_dict(),'pretrain_model/{}_encoder.pth'.format(expName))
        print("Model Has been saved !")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss",type=str,default='chamfer')
    parser.add_argument("--iters",type=int,default=100)
    parser.add_argument("--saved_model",type=str,default=None)
    parser.add_argument("--exp_name",type=str,default=None,required=True)
    parser.add_argument("--dataset",type=str,default='chopsticks')
    parser.add_argument("--freeze_encoder",action='store_true',default=False)

    args = parser.parse_args()

    main(
        args.loss,
        args.iters,
        args.saved_model,
        args.exp_name,
        args.dataset,
        args.freeze_encoder
    )

        