import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 


# Assume Pointcloud data is compressed as [idxs,N,6]
class PointCloudDataset(Dataset):
    def __init__(self,npz_file):
        pointclouds = np.load(npz_file)
        self.actions = pointclouds['action']
        self.state_x = pointclouds['before_x']
        self.state_v = pointclouds['before_v']
        self.state_F = pointclouds['before_F']
        self.state_C = pointclouds['before_C']
        self.target_x = pointclouds['after_x']
        self.n_particles = self.state_x.shape[1]
        self.n_actions = self.actions.shape[1]
        
    def __len__(self):
        return len(self.state_x)

    def __getitem__(self,idx):
        #idx = 0
        if torch.is_tensor(idx):
            idx = idx.to_list()
        state = [self.state_x[idx],self.state_v[idx],self.state_F[idx],self.state_C[idx]]
        target = [self.target_x[idx]]
        action = self.actions[idx]
        return state, target, action

class ChopSticksDataset(PointCloudDataset):
    def __init__(self):
        super(ChopSticksDataset,self).__init__('data/chopsticks.npz')

class RopeDataset(PointCloudDataset):
    def __init__(self):
        super(RopeDataset,self).__init__('data/rope.npz')

class TableDataset(PointCloudDataset):
    def __init__(self):
        super(TableDataset,self).__init__('data/table.npz')

class ChopsticksCFMDataset(Dataset):
    def __init__(self):
        pointclouds = np.load('data/chopsticks.npz')
        self.actions = torch.from_numpy(pointclouds['action']).float()
        self.state_x = torch.from_numpy(pointclouds['before_x']).float().permute(0,2,1)
        self.target_x = torch.from_numpy(pointclouds['after_x']).float().permute(0,1,3,2)
        self.n_particles = self.state_x.shape[1]
        self.n_actions = self.actions.shape[1]

    def __len__(self):
        return len(self.state_x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        state = self.state_x[idx]
        target = self.target_x[idx,0] # Only select the first frame.
        action = self.actions[idx,0]
        return state, target, action

if __name__ == '__main__':
    dataset = ChopsticksCFMDataset()
    dataloader = DataLoader(dataset,batch_size=4)
    for state,target,action in dataloader:
        print(state.shape)
        print(target.shape)
        print(action.shape)
        break