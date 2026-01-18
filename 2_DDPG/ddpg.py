import torch
from torch import Tensor
import torch.optim as optim

from gymnasium import Env
import numpy as np

from .policy_new import Policy
from .critic_new import Critic
from .replay_memory import ReplayMemory
from .utilities import freeze_module


class DDPG:
    def __init__(self, env: Env, *, 
                 batch_size: int = 100, gamma: float = 0.99, 
                 lr: float=0.005, soft_delay: int = 2):
        self._policy = Policy(
            env.observation_space,
            env.action_space,
            env.action_space,
            env.action_space, 
        ) 
        self._critic = Critic(
            env.observation_space,
            env.action_space,
            env.action_space,
            env.action_space, 
        )

        self._batch_size = batch_size
        self._gamma = gamma

        self._log_alpha = torch.tensor(0.0, requires_grad=True)
        self._target_entropy =  -np.prod(env.action_space.shape)

        self._soft_delay = soft_delay

        self._critic_optimizer = optim.AdamW(self._critic.parameters(), lr=lr) # target networks update differently
        self._actor_optimizer = optim.AdamW(self._policy.parameters(), lr=lr) # target networks update differently
        self._alpha_optimizer = optim.AdamW(self._policy.parameters(), lr=lr) # target networks update differently
        self._huber_loss = torch.nn.SmoothL1Loss()

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    def _update_alpha(self, log_pi: Tensor):
        loss = -(self.log_alpha * (log_pi.detach() + self._target_entropy).detach()).mean()
        self._alpha_optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._alpha_optimizer.step()

    def get_action(self, obs: Tensor) -> np.ndarray:
        with torch.no_grad():
            action, _ = self._policy.get_action(obs)
        return action.to_numpy()

    def update(self, step: int, replay_memory: ReplayMemory):
        batch = replay_memory.sample(self._batch_size)

        critic_loss = self._update_critic(batch)
        actor_loss, log_pi = self._update_actor(batch)
        self._update_alpha(log_pi)

        if step % self._soft_delay:
            self._soft_update()

    def _soft_update(self):
        self._policy.soft_update()
        self._critic.soft_update()

    def _update_actor(self, batch):
        observations = batch.observations

        # we sample from the actor distribution for the action
        # note that the previous action forms part of the state, so we pass it along with the observation
        actions, log_pi = self._policy.get_action(observations, target=False)

        with freeze_module(self._critic):
            # note that the previous action forms part of the state, so we pass it along with the observation
            q = self._critic.get_value(observations, actions)
        
        loss = (self._alpha.detach() * log_pi - q).mean()
        
        self._actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
        self._actor_optimizer.step()

        return loss.item(), log_pi

    def _update_critic(self, batch):
        observations = batch.observations
        next_observations = batch.next_observations
        rewards = batch.rewards
        actions = batch.actions

        q = self._critic.get_value(observations, actions)

        backup = self._get_backup(next_observations, rewards)

        loss: torch.Tensor = self._huber_loss(q, backup).mean()

        self._critic_optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._critic_optimizer.step()

        return loss.item()

    def _get_backup(self, next_observations, rewards):
        with torch.no_grad():
            a_p1, lop_pi_p1 = self._policy.get_action(next_observations, target=True)
            q_p1 = self._critic.get_value(next_observations, a_p1, target=True)
            backup = rewards + self._gamma * q_p1 - self._alpha.detach() * lop_pi_p1
        return backup
    