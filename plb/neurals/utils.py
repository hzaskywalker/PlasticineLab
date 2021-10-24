import torch
import torch.nn as nn

class MIMOSequential(nn.Module):
    def __init__(self,*args):
        super(MIMOSequential,self).__init__()
        self.list = nn.ModuleList()
        for arg in args:
            self.list.append(arg)

    def __getitem__(self,idx):
        return self.list[idx]
    
    def forward(self,*args):
        output = args
        for module in self.list:
            output = module(*output)
        return output
        