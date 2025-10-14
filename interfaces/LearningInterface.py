class LearningInterface:
    def update_value(self, s, s2, action, action2, reward):
        """Update the state action value given the current state and next state"""
        raise NotImplementedError

    def get_error(self, s: int, a: int, s2: int, r: float) -> float:
        """The error used when updating the value"""
        raise NotImplementedError

    def get_best_action(self, s) -> int:
        """For a state s, return the most probable action"""
        raise NotImplementedError

    def end(self, s_end: int, action: int):
        """Final update to values after the terminal state is found"""
        pass

    def reset(self):
        """Reset interal values for new episode"""
        pass
