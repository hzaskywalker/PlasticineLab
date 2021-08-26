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
        self.state_p = pointclouds['before_p']
        self.target_x = pointclouds['after_x']
        
    def __len__(self):
        return len(self.state_x)

    def __getitem__(self,idx):
        #idx = 2
        if torch.is_tensor(idx):
            idx = idx.to_list()
        state = [self.state_x[idx],
                 self.state_v[idx],
                 self.state_F[idx],
                 self.state_C[idx],
                 self.state_p[idx]]
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

if __name__ == '__main__':
    import time
    dataset = ChopSticksDataset()
    dataloader = DataLoader(dataset,batch_size=100,num_workers=5)
    start_time = time.time()
    for state,target, action in dataloader:
        print(len(state))
        print(len(target))
        break
    print(time.time()-start_time)