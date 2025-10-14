import numpy as np

class PolicyInterface:
    def __init__(self):
        self._policy = np.zeros(1)
        raise NotImplementedError

    def use_policy(self, s: int) -> int:
        """Pick an action given the sate"""
        raise NotImplementedError

    def update_policy(self, s: int, action: int):
        """Given the state and action, update the policy"""
        raise NotImplementedError

    def get_most_likely_action(self, s: int):
        """Given the state, return the most likely action, is not stochastic"""
        return int(np.argmax(self._policy[s]))
