import torch as T
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=[8, 8], **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Shared MLP
        self.mlp_extractor = CustomMLP(obs_dim, net_arch)
        last_layer_dim = net_arch[-1]

        # Actor
        self.action_net = nn.Linear(last_layer_dim, act_dim)
        self.log_std = nn.Parameter(T.zeros(act_dim))

        # Critic
        self.value_net = nn.Linear(last_layer_dim, 1)

        # Action distribution
        self.dist = DiagGaussianDistribution(act_dim)

        self._initialize_weights()

    def forward(self, obs, deterministic=False):
        features = self.mlp_extractor(obs)
        mean = T.tanh(self.action_net(features))
        std = T.clamp(T.exp(self.log_std), min=1e-3, max=1.0)
        dist = Normal(mean, std)

        actions = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        values = self.value_net(features)
        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        features = self.mlp_extractor(obs)
        mean = T.tanh(self.action_net(features))
        return mean if deterministic else Normal(mean, T.exp(self.log_std)).sample()

    def evaluate_actions(self, obs, actions):
        features = self.mlp_extractor(obs)
        mean = T.tanh(self.action_net(features))
        std = T.clamp(T.exp(self.log_std), min=1e-3, max=1.0)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        values = self.value_net(features)
        return values, log_prob, entropy
    
    def _initialize_weights(self):
        # Example: orthogonal init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

