from dqn import DQN
import torch.nn as nn


class TargetNetworkUpdate:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, target_net: nn.Module, policy_net: nn.Module):
        raise NotImplementedError


class SoftTargetNetworkUpdate(TargetNetworkUpdate):
    def __init__(self, tau: float):
        assert 0 < tau 
        assert 1 > tau
        self.tau = tau 

    def update(self, target_net: nn.Module, policy_net: nn.Module):
        # Note: Human-level control through deep reinforcement learning does a hard overwriting of the weights every C steps
        # This is a soft step update
        #   θ′ ← τ θ + (1 −τ )θ′
        # preventing large jumps in the target and a smoother training process(apparently)


        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        target_net.load_state_dict(target_net_state_dict)