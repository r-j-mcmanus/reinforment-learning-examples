import numpy as np


class ModelInterface:
    """The Model is solely responsible for providing a predicted reward given a state and action, where predit and update are to be overidden by concrete algorithms"""
    _REWARD = 0
    _STATE = 1

    def __init__(self, n_states, n_actions):
        self._model = np.zeros((n_states, n_actions, 2))

    def predict(self, state: int, action: int)  -> tuple[float, int]:
        """Use the model to predict the outcome of the action in the current state"""
        raise NotImplementedError

    def update(self, state: int, action: int, state2: int, reward: float) -> None:
        """Update the model with the results of an action in a state with its resulting state and reward"""
        raise NotImplementedError

    def get_preceding_states(self, s1: int) -> list[tuple[int, int, float]]:
        """Returns state, actions and their model rewards that lead to the state s1"""
        # Find matching indices
        matching_indices = np.where(self._model[:, :, self._STATE] == s1)

        # Extract triplets
        return [(s, a, self.predict(s, a)[0]) for s, a in zip(*matching_indices)]
