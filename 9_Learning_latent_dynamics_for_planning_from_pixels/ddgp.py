import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from constants import Constants
from episode_memory import EpisodeMemory, Transition
from rssm import RSSM
from state import State

# Basic Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        width = Constants.Common.MLP_width
        self.l1 = nn.Linear(state_dim, width)
        self.l2 = nn.Linear(width, width)
        # Output mean and log_std
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)
        self.max_action = max_action

        self.LOG_STD_MIN = -26
        self.LOG_STD_MAX = 2

    def forward(self, x, deterministic=False, reparameterize=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        mean: Tensor = self.mean(x)
        log_std: Tensor = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            z = mean
        else:
            # Reparameterization trick
            z = normal.rsample() if reparameterize else normal.sample()

        # Squash and scale
        action = self.max_action * torch.tanh(z)

        # Log-prob with Tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)

        return action, log_prob, normal

def _orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization maintains stable variance through deep networks and works 
    very well with tanh / ReLU activations."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)


# Basic Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        width = Constants.Common.MLP_width
        self.l1 = nn.Linear(state_dim, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 1)

    def forward(self, state) -> Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x).squeeze(dim=-1)

class DDPG:
    def __init__(self, action_dim, max_action,
                c_lr=Constants.Behaviors.critic_learning_rate, 
                a_lr=Constants.Behaviors.actor_learning_rate, 
                gamma=Constants.Behaviors.discount_factor, 
                tau=Constants.Behaviors.tau, 
                state_dim = Constants.World.latent_state_dimension):
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim)
        
        self.actor.apply(lambda l: _orthogonal_init(l, gain=0.01))
        self.critic.apply(lambda l: _orthogonal_init(l, gain=1.0))

        self.critic_target = Critic(state_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=c_lr)

        self.critic_loss_foo = F.mse_loss

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state, reparameterize=False) -> Tensor:
        if len(state.shape) == 1:
            state = torch.FloatTensor(state.reshape(1, -1))
        action, log_prob, normal = self.actor(state, reparameterize=reparameterize)
        return action.detach()

    def train(self, episode_memory: EpisodeMemory, rssm: RSSM, df: pd.DataFrame, batch_size: int = Constants.Behaviors.trajectory_count):
        transitions = episode_memory.sample(batch_size)

        h = torch.zeros([transitions.state.shape[0], Constants.World.hidden_state_dimension])
        obs = transitions.state[:,0,:]
        actions = transitions.action[:,0,:]
        next_obs = transitions.state[:,1,:]
        rewards = transitions.reward[:,0,:]
        
        with torch.no_grad():
            states = rssm.representation(obs, h)
            next_states = rssm.representation(next_obs, h)

        critic_loss = self.train_critic(states.sample, actions, rewards, next_states.sample)
        actor_loss, reinforce_loss, reparam_loss, entropy_loss = self.train_actor(states)

        self.soft_update()
        # for logging 
        row = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'reinforce_loss': reinforce_loss,
            'reparam_loss': reparam_loss,
            'entropy_loss': entropy_loss
        }
        df = pd.concat([df, pd.DataFrame([row])])

    def soft_update(self):
        """Soft update target networks"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_actor(self, states: State) -> tuple[float, float, float, float]:
        # Actor loss eq.6 2010.02193
        rho = Constants.Behaviors.actor_gradient_mixing
        
        states_sample = states.sample

        # reinforce_loss
        action_sf, log_prob_sf, dist_sf = self.actor(states_sample, reparameterize=False)
        q_value = self.critic(states_sample, action_sf).detach()
        reinforce_loss = -rho * (log_prob_sf * q_value.detach()).mean()
        
        # reparam_loss
        action_rp, log_prob_rp, dist_rp = self.actor(states_sample, reparameterize=True)
        q_value = self.critic(states_sample, action_rp)
        reparam_loss = -(1-rho) * (q_value.mean())

        # entropy loss
        entropy_loss = -Constants.Behaviors.actor_entropy_loss_scale * dist_rp.entropy().sum(dim=1).mean()

        actor_loss = reinforce_loss + reparam_loss + entropy_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), reinforce_loss.item(), reparam_loss.item(), entropy_loss.item()

    def train_critic(self, states: Tensor, actions: Tensor, rewards: Tensor, next_states: Tensor) -> float:
        # Critic loss eq.5 2010.02193
        with torch.no_grad():
            next_actions, _, _ = self.actor(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            #target = rewards + (1 - dones) * self.gamma * target_Q
            target = rewards + self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()