from interfaces.LearningInterface import LearningInterface


class PlannedLearningInterface(LearningInterface):
    """Interface for learning where a model is used for planning"""
    def direct_update(self, state: int, action: int, state2: int, reward):
        raise NotImplementedError