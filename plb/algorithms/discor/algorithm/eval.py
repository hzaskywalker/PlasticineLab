import os

from .sac import SAC
from ..network import GaussianPolicy
from ..utils import disable_gradients


class EvalAlgorithm(SAC):

    def __init__(self, state_dim, action_dim, device,
                 policy_hidden_units, GaussianPolicy):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._policy_net = GaussianPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
            ).eval().to(self._device)
        disable_gradients(self._policy_net)

    def load_models(self, save_dir):
        self._policy_net.load(os.path.join(save_dir, 'policy_net.pth'))
