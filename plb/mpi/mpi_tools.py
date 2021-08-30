from typing import Any, List, Union
import os, subprocess, sys

import torch

from mpi4py import MPI
import numpy as np
import pynvml

def best_mpi_subprocess_num(batch_size: int) -> int:
    """ Determine the most suitable number of sub processes

    The method will returns the minimum of the following:
    * batch size;
    * number of GPU cores, and
    * number of CPU cores. 

    :param batch_size: the size of each batch
    :return: the minimum of the above three
    """
    cpu_num = os.cpu_count()
    pynvml.nvmlInit()
    num_cuda = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()

    return min(batch_size, cpu_num, num_cuda)

def _abs_path_2_module_name(absPath:str) -> str:
    absPath = absPath.rstrip('.py').split('/')
    repoRoot = os.path.abspath('.').split('/') # path to where the python -m is originally executed
    i = 0
    while i < min(len(absPath), len(repoRoot)):
        if absPath[i] == repoRoot[i]: i += 1
        else:                         break
    return '.'.join(absPath[i:])
    


def mpi_fork(n: int, bind_to_core: bool=False) -> None:
    """
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name`_.
    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n (int): Number of process to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += ['python3', '-m', _abs_path_2_module_name(sys.argv[0])] + sys.argv[1:]
        subprocess.check_call(args, env=env)
        sys.exit()

def msg(m, string='') -> None:
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))

def proc_id() -> int:
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def num_procs() -> int:
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

def broadcast(x: Union[torch.Tensor, np.ndarray, Any], root: int=0) -> None:
    MPI.COMM_WORLD.Bcast(x, root=root)

def mpi_op(x: Union[torch.Tensor, np.ndarray, List, Any], op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    if isinstance(x, torch.Tensor):
        x = np.asarray(x.cpu(), dtype=np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x: Union[torch.Tensor, np.ndarray, List, Any]):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x: Union[torch.Tensor, np.ndarray, List, Any]):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x: List, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std