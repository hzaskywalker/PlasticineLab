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
        self.n_particles = self.state_x.shape[1]
        self.n_actions = self.actions.shape[1]
        
        np.random.seed(10)
        
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
        return state, target, action, idx

    # All methods regardless of dataset or subdataset requires to set loss to here
    def recordLoss(self,idxs,losses):
        self.loss_table[idxs] = losses

    # Weighted Sampling from subset, no allow repeat?
    # Subset doesn't have weight since there is no need for more than one subsampling
    def getSubset(self,size,replace=False):
        self.loss_table += 1e-5
        subset_idx = np.random.choice(len(self.state_x),p=self.loss_table/self.loss_table.sum(),size=size,replace=replace)
        sub_state_x = self.state_x[subset_idx]
        sub_state_v = self.state_v[subset_idx]
        sub_state_F = self.state_F[subset_idx]
        sub_state_C = self.state_C[subset_idx]
        sub_state_p = self.state_p[subset_idx]
        sub_target_x = self.target_x[subset_idx]
        sub_actions = self.actions[subset_idx]
        subdataset = PointCloudSubDataset(subset_idx,
                                          sub_state_x,
                                          sub_state_v,
                                          sub_state_F,
                                          sub_state_C,
                                          sub_state_p,
                                          sub_target_x,
                                          sub_actions)
        return subdataset
        

class PointCloudSubDataset(Dataset):
    def __init__(self,idxs,state_x,state_v,state_F,state_C,state_p,target_x,actions):
        self.state_x = state_x
        self.state_v = state_v
        self.state_F = state_F
        self.state_C = state_C
        self.state_p = state_p
        self.target_x = target_x
        self.actions = actions
        self.original_idxs = idxs
    
    def __len__(self):
        return len(self.state_x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        state = [self.state_x[idx],
                 self.state_v[idx],
                 self.state_F[idx],
                 self.state_C[idx],
                 self.state_p[idx]]
        target = [self.target_x[idx]]
        action = self.actions[idx]
        original_idx = self.original_idxs[idx]
        return state, target, action, original_idx
    
class PointCloudAEDataset(Dataset):
    def __init__(self,npz_file):
        pointclouds = np.load(npz_file)
        self.x = torch.from_numpy(pointclouds['before_x']).permute(0,2,1)
        self.n_particles = self.x.shape[2]

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.x[idx]

class ChopSticksDataset(PointCloudDataset):
    def __init__(self):
        super(ChopSticksDataset,self).__init__('data/old_dataset/chopsticks.npz')

class RopeDataset(PointCloudDataset):
    def __init__(self):
        super(RopeDataset,self).__init__('data/old_dataset/rope.npz')

class TorusDataset(PointCloudDataset):
    def __init__(self):
        super(TorusDataset,self).__init__('data/torus-v1.npz')

class WriterDataset(PointCloudDataset):
    def __init__(self):
        super(WriterDataset,self).__init__("data/writer-v1.npz")


class CFMDataset(Dataset):
    def __init__(self, env:str):
        pointclouds = np.load(f'data/{env}.npz')
        self.actions = torch.from_numpy(pointclouds['action']).float()
        self.state_x = torch.from_numpy(pointclouds['before_x']).float().permute(0,2,1)
        self.target_x = torch.from_numpy(pointclouds['after_x']).float().permute(0,1,3,2)
        self.n_particles = self.state_x.shape[-1]
        self.n_actions = self.actions.shape[-1]

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
    import time
    dataset = ChopSticksDataset()
    dataloader = DataLoader(dataset,batch_size=100,num_workers=5)
    start_time = time.time()
    for state,target, action in dataloader:
        print(len(state))
        print(len(target))
        break
    print(time.time()-start_time)
