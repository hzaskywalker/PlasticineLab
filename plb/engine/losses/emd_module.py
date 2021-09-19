import torch
import torch.nn as nn
import emd
from torch.autograd import Function

class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m)
        assert(xyz1.size()[0] == xyz2.size()[0])
        assert(n % 1024 == 0)
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None

class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)

emd_function = emdModule()

# Assume only one frame input
def compute_emd(gt,output,iters):
    if not isinstance(output,torch.Tensor):
        output = torch.from_numpy(output).float().cuda()
    if not isinstance(gt,torch.Tensor):
        gt = torch.from_numpy(gt).float().cuda()
    s = output.shape
    output = output.view(1,s[0],s[1])
    gt = gt.view(1,s[0],s[1])
    emd_loss,assignment = emd_function(gt,output,0.02,iters)
    loss = torch.sqrt(emd_loss).mean()*10
    return loss,assignment[0]