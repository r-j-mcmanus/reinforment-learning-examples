import torch
from torch import Tensor
import torch.optim as optim

import numpy as np

from policy import Policy
from critic import Critic
from replay_memory import ReplayMemory, Transition


class CrossQ:
    """Implementing 1812.05905"""
    def __init__(self, observation_dim: int, action_dim: int, action_max: float, action_min: float,
                 *,
                 n_steps: int = 10,
                 batch_size: int = 100, gamma: float = 0.99, 
                 lr: float=0.005, soft_delay: int = 2):
        self._policy = Policy(
            observation_dim,
            action_dim,
            action_max,
            action_min, 
        ) 
        self._critic = Critic(
            observation_dim,
            action_dim
        )
        
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._gamma = gamma

        self._log_alpha = torch.tensor(-5.0, requires_grad=True)
        self._target_entropy =  -action_dim # see table 1 in 1812.05905

        self._soft_delay = soft_delay

        self._alpha_optimizer = optim.AdamW([self._log_alpha], lr=lr)

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    def _update_alpha(self, batch: Transition):
        _, log_pi = self._policy.get_action(batch.state)

        # Alpha loss
        alpha_loss = -(self._log_alpha * (log_pi.detach() + self._target_entropy)).mean()

        # Optimize log_alpha
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()

    def get_action(self, obs: Tensor, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action, _ = self._policy.get_action(obs, deterministic)
        return action.cpu().numpy()[0]

    def update(self, replay_memory: ReplayMemory):
        for step in range(self._n_steps):
            batch = replay_memory.sample(self._batch_size)

            critic_loss = self._update_critic(batch)
            actor_loss = self._update_actor(batch)
            self._update_alpha(batch)

    def _update_actor(self, batch: Transition) -> float:
        observations: Tensor = batch.state

        # we sample from the actor distribution for the action
        # note that the previous action forms part of the state, so we pass it along with the observation
        actions, log_pi = self._policy.get_action(observations)

        q = self._critic.get_value(observations, actions)
        
        loss: Tensor = (self._alpha.detach() * log_pi - q).mean()
        
        self._policy.update(loss)

        return loss.item()

    def _update_critic(self, batch: Transition) -> float:
        observations: Tensor = batch.state
        observations_p1: Tensor = batch.next_state
        rewards: Tensor = batch.reward
        actions: Tensor = batch.action
        done: Tensor = batch.done


        with torch.no_grad():
            actions_p1, lop_pi_p1 = self._policy.get_action(observations_p1)

        loss = self._critic.update(
            observations, observations_p1, rewards, actions, actions_p1, 
            lop_pi_p1, done, self._alpha.detach(), self._batch_size
        )

        return loss
