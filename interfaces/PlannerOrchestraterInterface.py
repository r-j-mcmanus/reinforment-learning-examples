from interfaces.ModelInterface import ModelInterface
from interfaces.PlannedLearningInterface import PlannedLearningInterface

class PlannerOrchestraterInterface:
    """Allows for the leaner to have different planning methods using a model"""

    def __init__(self, model: ModelInterface) -> None:
        self._model = model

    def plan(self, learner: PlannedLearningInterface) -> None:
        raise NotImplementedError

    def update_model(self, state: int, action: int, state2: int, reward: float) -> None:
        self._model.update(state, action, state2, reward)