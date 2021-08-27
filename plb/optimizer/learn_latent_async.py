# UNUSED
# TODO: Delete the file once MPI works
import multiprocessing
import os
from typing import Tuple, Union, List, Any

import numpy as np
import pynvml
import torch

from plb.engine.taichi_env import TaichiEnv
from plb.optimizer.learn_latent import Solver


def device_selector()->torch.device:
    """Select the CUDA device
    
    The one with maximum free memory will be picked

    :return: torch device handler to the selected CUDA device
    """
    pynvml.nvmlInit()
    num_cuda = pynvml.nvmlDeviceGetCount()
    max_device_idx, max_device_mem = 0
    for i in range(num_cuda):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlUnitGetHandleByIndex(i))
        if meminfo.free > max_device_mem:
            max_device_idx, max_device_mem = i, meminfo.free
    pynvml.nvmlShutdown()

    return torch.device("cuda:%d"%(max_device_idx))

class AsyncSolver(Solver):
    """ Run the solver to learn latent states in an asyn way, using process pool

    :param env: the Taichi enviornment
    :param model: a torch.nn.Module
    :param logger: logger to logging the performance and runtime data for EACH solver
    :param cfg: configuration
    :decay_factor: a value to detemine the decay
    """
    def __init__(self, env: TaichiEnv, model, logger, cfg, decay_factor, steps, **kwargs):
        super().__init__(env, model, optimizer=None, logger=logger, cfg=cfg, decay_factor=decay_factor, steps=steps, **kwargs)
    
    class MessagePack:
        """ Package as the message to be transmitted through master and slaves

        :param state: corresponding to the state parameter in solve.solve_multistep
        :param actions: corresponding to the actions parameter in solve.solve_multistep
        :param target: corresponding to the target parameter in solve.solve_multistep
        """
        def __init__(self, state, actions, target) -> None:
            self._in_state   = state
            self._in_actions = actions
            self._in_target  = target

            self._out_grads  = []
            self._out_states = []
            self._out_losses = []
            self._last_loss  = None


        def append_out_grads(self, to:List):
            """ Append the grads returned from an async solve_multisteps to the gradient buffer

            This is thread-unsafe, so one has to use this in the main thread

            :param to: to which buffer the gradient should be appended to
            """
            if self._out_grads is not None:
                to.extend(self._out_grads)

        def append_out_states(self, to:List):
            """ Append the states returned from an async solve_multisteps to the state buffer

            This is thread-unsafe, so one has to use this in the main thread

            :param to: to which buffer the states should be appended to
            """
            if self._out_states is not None:
                to.extend(self._out_states)

        def append_out_losses(self, to:list):
            """ Append the losses returned from an async solve_multisteps to the loss buffer

            This is thread-unsafe, so one has to use this in the main thread

            :param to: to which buffer the losses should be appended to
            """
            if self._out_losses is not None:
                to.extend(self._out_losses)

        @property
        def multistep_results(self) -> Tuple[torch.Tensor, torch.Tensor, Union[np.ndarray, torch.Tensor], Any]:
            """ The unpacked results from this message object

            A tuple of four elements: 
            * the next state, to be appended to the state buffer;
            * the gradient, to be appended to the gradient buffer;
            * the loss, to be appended to the loss buffer, and
            * the last loss, which is the returned value of the SYNC version of solver.solve_multisteps
            """
            return (self._out_states, self._out_grads, self._out_losses, self._last_loss)
            

    def solve_multistep_async(self, messagePack:"MessagePack") -> "MessagePack":
        """ Solve multipstep in an async way

        This method CAN be called parallel

        :param messagePack: the object containing the parameters for the
            SYNC version of solver.solve_multisteps and placeholder for
            results
        :return: exactly the same object as the messagePack parameter,
            with its placeholder for the results being updated
        """
        messagePack._last_loss = self.solve_multistep(
            state        = messagePack._in_state,
            actions      = messagePack._in_actions,
            targets      = messagePack._in_target,
            grad_buffer  = messagePack._out_grads,
            state_buffer = messagePack._out_states,
            loss_buffer  = messagePack._out_losses,
            local_device = device_selector()
        )

        return messagePack

class SlaveManager:
    """ Manage all the slave processes

    NEVER instantiate this class unless using the factory method
    """
    def __init__(self, solver: AsyncSolver) -> None:
        self.pool = multiprocessing.Pool(os.cpu_count() - 1)
        self.solver = solver

    @classmethod
    def factory(cls, env: TaichiEnv, model, logger, cfg, decay_factor, steps, **kwargs):
        """ Create an AsyncSolver and encapsulate it into a SlaveManager

        The parameters are the arguments to construct a AsyncSolver. This
        method will make necessary copies of the input arguments to avoid
        parallel issues. 
        """
        #TODO: copy and load stuff
        return SlaveManager(AsyncSolver(env, model, logger, cfg, decay_factor, steps, **kwargs))


    def execute_batch(self, batch: List[AsyncSolver.MessagePack], 
            grad_buffer: List, state_buffer: List, loss_buffer: List):
        """ Execute solver.solve_multistep in parallel

        Use a process pool to execute the solver in batch. This
        will block the current thread until the batch has been
        completed totally. 

        :param batch: a list of messages, containing the states, 
            the actions and the target
        :param grad_buffer: buffer for grads returned from solvers
        :param state_buffer: buffer for states returned from solvers
        :param loss_buffer: buffer for losses returned from solvers
        """
        results: List[multiprocessing.pool.AsyncResult] = []
        for bm in batch:
            results.append(self.pool.apply_async(self.solver.solve_multistep_async, args=(bm,)))

        for result in results: result.wait()

        for i, result in enumerate(results):
            bmResult = result.get()
            if bmResult is not None and isinstance(bmResult, AsyncSolver.MessagePack):
                batch[i] = result.get()
                batch[i].append_out_grads(to=grad_buffer)
                batch[i].append_out_states(to=state_buffer)
                batch[i].append_out_losses(to=loss_buffer)

        