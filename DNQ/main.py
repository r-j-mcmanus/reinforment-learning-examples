import random
import gymnasium as gym
from itertools import count
from typing import Type, Self

import torch
import torch.optim as optim
import torch.nn as nn

from DNQ.dqn import DQN
from replay_memory import Memory, ReplayMemory
from policy import Policy, EpsGreedyPolicy
from optimise import apply_q_learning_step
from constants import *
from update import TargetNetworkUpdate, SoftTargetNetworkUpdate

# ensure reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# see Human-level control through deep reinforcement learning
# and Playing Atari with Deep Reinforcement Learning
# for initial implementation details

class ReinforcementLearner:
    def __init__(self):
        # maybe set global vars here?
        pass

    def set_memory(self, memory_class: Type[Memory]) -> Self:
        self.memory_class = memory_class
        return self
        
    def set_policy(self, policy_class: Type[Policy]) -> Self:
        self.policy_class = policy_class
        return self
        
    def set_target_updater(self, target_updater_class: Type[TargetNetworkUpdate]) -> Self:
        self.target_updater_class = target_updater_class
        return self
        
    def set_dqn(self, dqn_class: Type[nn.Module]) -> Self:
        self.dqn_class = dqn_class
        return self

    def set_optimiser(self, optimiser_class: Type[optim.Optimizer]) -> Self:
        self.optimiser_class = optimiser_class
        return self

    def set_env(self, env) -> Self:
        self.env = env
        return self

    def build(self) -> Self:
        # TODO push build into their own classes 

        state, _ = self.env.reset()
        n_observations = len(state)
        n_actions = self.env.action_space.n

        self.memory = self.memory_class(10000) # previous transitions stored for batch sampling
        
        # we use both a target net and a policy net to stabalise training
        # this is as when training the policy network TODO
        self.policy_net = self.dqn_class(n_observations, n_actions) # learns and updates during training
        self.target_net = self.dqn_class(n_observations, n_actions) # stable targets for the Q-learning update
        self.target_net.load_state_dict(self.policy_net.state_dict()) # insure initially equal

        # for selecting next transition
        self.policy = self.policy_class(self.policy_net) 

        # AdamW (a better version of Adam) is a modified gradient decent algorithm
        self.optimizer = self.optimiser_class(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

        self.target_updater = self.target_updater_class(TAU)

        return self
    
    def run(self):
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50


        # tracking how long the episode lasted
        episode_durations = []

                    
        for i_episode in range(num_episodes):

            # randomly initialise the enviroment, get the corrisponding state
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # unsqueeze inserts a dimention into a tensor
            # e.g torch.unsqueeze(x, 1)
            # tensor([[ 1],
            #         [ 2],
            #         [ 3],
            #         [ 4]])

            for t in count():
                action = self.policy.select_action(self.env, state)

                observation, reward, terminated, truncated, _ = self.env.step(action.item()) # gymnasium response to the action
                # truncated = if the episode ended due to a time limit, step limit, or other artificial cutoff.
                # terminated = if the episode ended because the agent reached a terminal state

                reward = torch.Tensor([reward], device=DEVICE)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state for the next loop
                state = next_state

                # Perform one step of the optimization (on the policy network)
                apply_q_learning_step(self.memory, self.policy_net, self.target_net, self.optimizer)

                # Soft update of the target network's weights
                self.target_updater.update(self.target_net, self.policy_net)

                if done:
                    episode_durations.append(t + 1)
                    # plot_durations()
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    ReinforcementLearner() \
        .set_env(env) \
        .set_memory(ReplayMemory) \
        .set_policy(EpsGreedyPolicy) \
        .set_target_updater(SoftTargetNetworkUpdate) \
        .set_dqn(DQN) \
        .set_optimiser(optim.AdamW) \
        .run()
