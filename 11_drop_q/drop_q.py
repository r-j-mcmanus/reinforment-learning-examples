import torch
from torch import Tensor
import torch.optim as optim

import numpy as np

from policy import Policy
from critic import Critic
from replay_memory import ReplayMemory, Transition


class DropQ:
    """Implementing 2110.02034"""
    def __init__(self, observation_dim: int, action_dim: int, action_max: float, action_min: float,
                 *,
                 update_to_data: int = 10,
                 batch_size: int = 200, 
                 lr: float=0.03, 
                 policy_delay: int = 2):
        self._policy = Policy(
            observation_dim,
            action_dim,
            action_max,
            action_min,
            lr
        ) 
        self._critic = Critic(
            observation_dim,
            action_dim,
            lr
        )
        
        self._batch_size = batch_size

        self._log_alpha = torch.tensor(-5.0, requires_grad=True)
        self._target_entropy = -action_dim # see table 1 in 1812.05905

        self._alpha_optimizer = optim.AdamW([self._log_alpha], lr=lr)

        self._policy_delay = policy_delay
        self._policy_count = 0

        self._remaining_updates = 0 
        self._update_to_data = update_to_data 

    @property
    def alpha(self):
        return self._log_alpha.detach().exp()

    def _update_alpha(self, batch: Transition):
        _, log_pi = self._policy.get_action(batch.state)

        # Alpha loss
        alpha_loss = - (self._log_alpha * (log_pi.detach() + self._target_entropy)).mean()

        # Optimize log_alpha
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()

    def get_action(self, obs: Tensor, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action, _ = self._policy.get_action(obs, deterministic)
        return action.cpu().numpy()[0]

    def update(self, replay_memory: ReplayMemory):
        self._remaining_updates += self._update_to_data
        self._policy_count += 1
        
        while self._remaining_updates > 0:
            self._remaining_updates -= 1

            batch = replay_memory.sample(self._batch_size)

            critic_loss = self._update_critic(batch)
        
            actor_loss = 0
            if self._policy_count % self._policy_delay == 0:
                actor_loss = self._update_actor(batch)
                self._update_alpha(batch)
                self._critic.soft_update()

        return critic_loss, actor_loss

    def _update_actor(self, batch: Transition) -> float:
        observations: Tensor = batch.state

        actions, log_pi = self._policy.get_action(observations)

        q = self._critic.get_value(observations, actions)
        
        loss: Tensor = (self.alpha * log_pi - q).mean()
        
        self._policy.update(loss)

        return loss.item()

    def _update_critic(self, batch: Transition) -> float:
        observations: Tensor = batch.state
        observations_p1: Tensor = batch.next_state
        rewards: Tensor = batch.reward
        actions: Tensor = batch.action
        done: Tensor = batch.done

        with torch.no_grad():
            actions_p1, log_pi_p1 = self._policy.get_action(observations_p1)

        loss = self._critic.update(
            observations, observations_p1, rewards, actions, actions_p1, 
            log_pi_p1, done, self.alpha, self._batch_size
        )

        return loss
