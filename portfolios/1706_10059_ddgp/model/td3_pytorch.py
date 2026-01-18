from itertools import chain

from environment import AssetTrainingEnvironment

import torch
import torch.nn as nn
import torch.optim as optim

from model.convolution_actor import ConvolutionActorNetwork    
from model.convolution_critic import ConvolutionCriticNetwork  
from model.replay_buffer import ReplayBuffer, Sample 


class Actor(nn.Module):

    def __init__(self, 
                 feature_dim: int, history_len: int,
                 lr: float = 0.001, polyak: float = 0.005):
        """
        Use the policy to move in the environment.
        Use the target for updating the critic.
        """
        super().__init__()
        self.policy = ConvolutionActorNetwork(feature_dim, history_len)
        self.target = ConvolutionActorNetwork(feature_dim, history_len)

        self.policy.load_state_dict(self.target.state_dict()) # insure initially equal

        self._optimizer = optim.AdamW(self.policy.parameters(), lr=lr) # target networks update differently

        self._polyak = polyak

    def forward(self, obs: torch.Tensor, a_m1: torch.Tensor, use_target=False):
        if len(a_m1.shape)==1:
            a_m1 = a_m1.unsqueeze(dim=0)
        return self.target(obs, a_m1) if use_target else self.policy(obs, a_m1)

    def update(self, loss: torch.Tensor):
        self._optimizer.zero_grad()
        loss.backward() 
        self._optimizer.step()

    def soft_update(self):
        stabilising_net_state_dict = self.policy.state_dict()
        target_net_state_dict = self.target.state_dict()
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*self._polyak + target_net_state_dict[key]*(1-self._polyak)
        self.target.load_state_dict(target_net_state_dict)


class Critic(nn.Module):
    def __init__(self, 
                feature_dim: int, history_len: int, n_assets: int,
                 lr: float = 0.001, polyak: float = 0.005):
        """
        Use the target when calculating the backup.
        Use the Q function for updating the policy (by convention, use Q_1).
        """
        super().__init__()
        self.Q_1 = ConvolutionCriticNetwork(feature_dim, history_len, n_assets)
        self.Q_2 = ConvolutionCriticNetwork(feature_dim, history_len, n_assets)
        self.target_1 = ConvolutionCriticNetwork(feature_dim, history_len, n_assets)
        self.target_2 = ConvolutionCriticNetwork(feature_dim, history_len, n_assets)
        
        self.Q_1.load_state_dict(self.target_1.state_dict()) # insure initially equal        
        self.Q_2.load_state_dict(self.target_2.state_dict())

        # target networks update differently
        params = chain(self.Q_1.parameters(), self.Q_2.parameters()) 
        self._optimizer = optim.AdamW(params, lr=lr)
        self._criterion = nn.SmoothL1Loss()

        self._polyak = polyak

    def update(self, obs, a_m1, a, backup):
        loss_1 = self._criterion(self.Q_1(obs, a_m1, a), backup).mean()
        loss_2 = self._criterion(self.Q_2(obs, a_m1, a), backup).mean()
        loss: torch.Tensor = loss_1 + loss_2
        self._optimizer.zero_grad()
        loss.backward() 
        self._optimizer.step()

    def soft_update(self):
        q_net_state_dict = self.Q_1.state_dict()
        target_net_state_dict = self.target_1.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*self._polyak + target_net_state_dict[key]*(1-self._polyak)
        self.target_1.load_state_dict(target_net_state_dict)
        
        q_net_state_dict = self.Q_2.state_dict()
        target_net_state_dict = self.target_2.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*self._polyak + target_net_state_dict[key]*(1-self._polyak)
        self.target_2.load_state_dict(target_net_state_dict)

    def target_min_forward(self, x, a_m1, a):
        """Take the min state action value from the two target networks"""
        out_1 = self.target_1(x,a_m1,a)
        out_2 = self.target_2(x,a_m1,a)
        return torch.min(out_1, out_2)


class TD3:
    def __init__(self, 
                 min_action: float, max_action: float, 
                 feature_dim: int, history_len: int, n_assets: int,
                 *,
                 gamma: float = 0.99, policy_delay: int = 2, 
                 batch_size: int = 100, noise_clip: float = 0.5, policy_noise: float = 0.1):
        self.actor = Actor(feature_dim, history_len)
        self._critic = Critic(feature_dim, history_len, n_assets)

        self._gamma = gamma
        self._policy_delay = policy_delay
        self._batch_size = batch_size
        self._noise_clip = noise_clip
        self._policy_noise = policy_noise
        self._min_action = min_action
        self._max_action = max_action

    def train(self, update_number: int, env: AssetTrainingEnvironment, replay_buffer: ReplayBuffer):
        """
        Args:
            update_number (int): How many batch updates to do

            replay_buffer (ReplayBuffer): Buffer storing previous environment interactions
        """
        for j in range(update_number):
            sample= replay_buffer.sample(self._batch_size)

            # obs shape (batch, symbols, history, features)
            obs = env.get_batch_sample(sample.obs_idxs)
            next_obs = env.get_batch_sample(sample.next_obs_idxs)

            with torch.no_grad():
                noise = (torch.randn_like(sample.actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip) 
                next_action = self.actor(next_obs, sample.actions, use_target=True)
                noisy_next_action = (next_action + noise)
                normed_noisy_next_action = noisy_next_action / noisy_next_action.sum(dim=-1, keepdim=True)
                min_q_target = self._critic.target_min_forward(next_obs, sample.actions, normed_noisy_next_action)
                backup = sample.rewards + self._gamma * min_q_target

            self._critic.update(obs, sample.prev_actions, sample.actions, backup)

            if (j + 1) % self._policy_delay == 0:
                loss = - self._critic.Q_1(obs, sample.prev_actions, self.actor(obs, sample.prev_actions)).mean() # - to assent the expected reward
                self.actor.update(loss)
                self.actor.soft_update()
                self._critic.soft_update()
