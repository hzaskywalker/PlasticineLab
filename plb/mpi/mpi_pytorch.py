from typing import Union, Generator
from numpy import ndarray
import torch
from torch import Tensor, nn
from .mpi_tools import broadcast, mpi_avg, num_procs, proc_id

def setup_pytorch_for_mpi() -> None:
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)

def batch_collate(*batchs: Tensor, toNumpy: bool = False) -> Generator[Union[Tensor, ndarray], None, None]:
    """ Select the ones for the current process from the batch of testing result

    Example, 4 procosse and the batch is of length 8, then
    batchPerProc = 2, so
    batch[0], batch[1] => proc No.0
    batch[2], batch[3] => proc No.1
    batch[4], batch[5] => proc No.2
    batch[6], batch[7] => proc No.3

    :param batch: the batch from which testing data are to be selected
    """
    rank, batchLen, procCnt = proc_id(), len(batchs[0]), num_procs()
    assert all(len(batch) == batchLen for batch in batchs), \
        "all batch must be of the same length, but the lengths " \
        + f"are {[len(batch) for batch in batchs]}"
    
    batchPerProc = max(batchLen // procCnt, 1)
    
    if batchPerProc == 1 and all(hasattr(batch, "squeeze") for batch in batchs):
        gen = (batch[rank % batchLen].squeeze() for batch in batchs)
    else:
        gen = (batch[(rank * batchPerProc) % batchLen : ((rank + 1) * batchPerProc) % batchLen] for batch in batchs)
    
    if toNumpy and all(isinstance(batch, Tensor) for batch in batchs):
        return (batch.numpy() for batch in gen)
    return gen

def mpi_avg_grads(module: nn.Module) -> None:
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.cpu().numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]

def sync_params(module: nn.Module) -> None:
    """ Sync all parameters of module across all MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.cpu().numpy()
        broadcast(p_numpy)