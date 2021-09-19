import torch
from torch import nn
from torch.distributions import Normal


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):
    assert isinstance(input_dim, int) and isinstance(output_dim, int), f"{input_dim}:{type(input_dim)}, {output_dim}:{type(output_dim)}"
    assert isinstance(hidden_units, list) or isinstance(hidden_units, list)

    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers).apply(initialize_weights_xavier)


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class PreprocessNet(BaseNetwork):
    def __init__(self, state_dim):
        super(PreprocessNet, self).__init__()
        self.state_dim = state_dim

        #self.p_net = create_linear_network(
        #    input_dim=state_dim,
        #    output_dim=128,
        #    hidden_units=[32, 64]
        #)
        #self.output_dim = 8
        #self.output_dim = 256
        self.output_dim = state_dim

    def forward(self, x):
        #x = x.reshape(x.shape[0], -1, self.state_dim)
        #x = self.p_net(x)
        #x = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=1)
        return x

class StateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()
        self.net = create_linear_network(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_units=hidden_units)

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))


class TwinnedStateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net1 = StateActionFunction(state_dim, action_dim, hidden_units)
        self.net2 = StateActionFunction(state_dim, action_dim, hidden_units)

    def forward(self, states, actions):
        #assert states.dim() == 2 and actions.dim() == 2

        #x = torch.cat([states, actions], dim=1)
        value1 = self.net1(states, actions)
        value2 = self.net2(states, actions)
        return value1, value2


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.preprocess = PreprocessNet(state_dim)

        self.net = create_linear_network(
            input_dim=self.preprocess.output_dim,
            output_dim=2*action_dim,
            hidden_units=hidden_units)

    def forward(self, states):
        assert states.dim() == 2, f"but get shape {states.shape}"

        states = self.preprocess(states)

        # Calculate means and stds of actions.
        #print([i for i in self.net.parameters()])
        #print(self.net(states))
        #print(self.net(states).mean())
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        stds = log_stds.exp_()

        # Gaussian distributions.
        normals = Normal(means, stds)

        # Sample actions.
        xs = normals.rsample()
        #print('xs', xs)
        actions = torch.tanh(xs)

        # Calculate entropies.
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
