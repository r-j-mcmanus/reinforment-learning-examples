import numpy as np

from typing import Literal, Callable
from collections import deque

from ActionIndex import ActionIndex
from interfaces.PlannerOrchestraterInterface import PlannerOrchestraterInterface
from interfaces.ModelInterface import ModelInterface
from interfaces.PolicyInterface import PolicyInterface
from interfaces.LearningInterface import LearningInterface
from interfaces.PlannedLearningInterface import PlannedLearningInterface
from interfaces.WorldInterface import WorldInterface

METHOD_TYPE = Literal["sarsa", "q_learning"]

N_ACTIONS = 4


class GridStateEncoder2D:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def encode(self, x: int, y: int) -> int:
        return y * self.width + x

    def decode(self, index: int) -> tuple[int, int]:
        return index % self.width, index // self.width


class GridStateEncoder3D:
    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth

    def encode(self, x: int, y: int, z: int) -> int:
        return z * self.width * self.height + y * self.width + x

    def decode(self, index: int) -> tuple[int, int, int]:
        x = index % self.width
        y = (index // self.width) % self.height
        z = index // (self.width * self.height)
        return x, y, z


class RandomPlanningOrchestrater(PlannerOrchestraterInterface):
    def __init__(self, model: ModelInterface, n: int):
        self._n = n
        self._model = model
        self._seen_state_action_pairs = set()

    def update_model(self, state: int, action: int, state2: int, reward: float) -> None:
        self._seen_state_action_pairs.add((state, action))
        self._model.update(state, action, state2, reward)

    def plan(self, learner: PlannedLearningInterface):
        counter = 0
        for s, a in self._seen_state_action_pairs:
            r, s2 = self._model.predict(s, a)
            learner.direct_update(s, a, s2, r)
            counter += 1
            if counter >= self._n:
                return


class PrioritizedSweepingPlanningOrchestrater(PlannerOrchestraterInterface):
    """Implements the Prioritized Sweeping planning strategy for model-based reinforcement learning"""

    def __init__(self, model: ModelInterface, theta: float):
        """
        Parameters:
                model (ModelInterface.py): The model used to predict state transitions and rewards.
                theta (float): The minimum TD error required to trigger planning. Smaller values lead to 
                            more aggressive planning; larger values make planning more selective
        """
        super().__init__(model)
        self._theta = theta
        self._s: int = None
        self._s2: int = None
        self._r: float = None
        self._a: int = None

    def update_model(self, state: int, action: int, state2: int, reward: float, *, max_depth=3) -> None:
        # store the current state we have updated
        self._s = state
        self._s2 = state2
        self._r = reward
        self._a = action
        self._model.update(state, action, state2, reward)
        self._max_depth = max_depth

    def plan(self, learner: PlannedLearningInterface):
        # check if the state we updated
        self._check_stored_state()
        # ensure the update was large enough to warent planning
        p = np.abs(learner.get_error(self._s, self._a, self._s2, self._r))
        if p > self._theta:
            pq = [(self._s, self._a, p)]
            depth = 0
            while len(pq) > 0:
                if depth > self._max_depth:
                    break
                s, a = self.max_priority(pq)
                r, s2 = self._model.predict(s, a)
                learner.direct_update(s, a, s2, r)
                pq.extend(self._get_large_error_preceding_states(learner, s))
                depth += 1
        self._reset_stored_state()

    def _get_large_error_preceding_states(self, learner: PlannedLearningInterface, s: int) -> list[
        tuple[int, int, float]]:
        pq = []
        for s0, a0, r0 in self._model.get_preceding_states(s):
            p = np.abs(learner.get_error(s0, a0, s, r0))
            if p > self._theta:
                pq.append((s0, a0, p))
        return pq

    @classmethod
    def max_priority(cls, pq: list[tuple[int, int, float]]) -> tuple[int, int]:
        """pq = list of (state, action, priority) tuples, returns (state, action)) with the larges priorety, removing it from the list"""
        pq.sort(key=lambda x: x[2], reverse=True)
        return pq.pop(0)[:2]  # Return (state, action)

    def _reset_stored_state(self):
        self._s = self._s2 = self._r = self._a = None

    def _check_stored_state(self):
        if None in (self._s, self._s2, self._r, self._a):
            raise ValueError("No stored state-action update available for planning.")


class StaticModel(ModelInterface):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)

    def predict(self, state: int, action: int) -> tuple[float, int]:
        return self._model[state, action, self._REWARD], int(self._model[state, action, self._STATE])

    def update(self, state: int, action: int, state2: int, reward):
        """Update the model with the results of using an action in a state with its resulting state and reward"""
        self._model[state, action, self._REWARD] = reward
        self._model[state, action, self._STATE] = state2


class DecayingModel(ModelInterface):
    """We include a bonus on the models reward that increses as time since the state action was used.
    This promotes use of this older state when Q is updated using the model. 
    If the enviroment is dynamic, this will allow for the model to discover changes."""

    def __init__(self, n_states: int, n_actions: int, k: float, bonus_function: Callable[[float], float] | None = None):
        """
        Parameters:
            n_states (int): Number of states in the environment.
            n_actions (int): Number of possible actions.
            k (float): Scaling factor for the exploration bonus.
            bonus_function (Callable[[float], float], optional): A monotonic function mapping time since last visit 
                to a bonus value. Defaults to sqrt(time).
        """
        super().__init__(n_states, n_actions)
        self._time_since_visit = np.zeros((n_states, n_actions))
        self._seen_state_action_pairs = set()
        self._k = k
        if isinstance(bonus_function, Callable):
            self.bonus_function = bonus_function
        else:
            self.bonus_function = self._default_bonus_function

    def predict(self, state: int, action: int) -> tuple[float, int]:
        r, s2 = self._model[state, action, self._REWARD], int(self._model[state, action, self._STATE])
        r += self._k * self.bonus_function(self._time_since_visit[state, action])
        return r, s2

    def update(self, state: int, action: int, state2: int, reward):
        self._time_since_visit = self._time_since_visit + 1
        self._time_since_visit[state, action] = 0
        self._model[state, action, self._REWARD] = reward
        self._model[state, action, self._STATE] = state2
        self._seen_state_action_pairs.add((state, action))

    def _default_bonus_function(self, t: float) -> float:
        return np.sqrt(t)


class EpsilonGreedyPolicy(PolicyInterface):
    def __init__(self, eps, n_states):
        """
        Parameters:
                eps (float): The exploration rate (0 ≤ eps ≤ 1). Higher values encourage more exploration.
                n_states (int): The number of states in the environment. Used to initialize the policy table.
        """
        self._policy = np.ones((n_states, N_ACTIONS)) / N_ACTIONS
        self.eps = eps

    def use_policy(self, s) -> int:
        action_prob = self._policy[s, :]
        return np.random.choice(N_ACTIONS, p=action_prob)

    def update_policy(self, s, action):
        self._policy[s, action] = 1 - self.eps + self.eps / N_ACTIONS
        for a in range(N_ACTIONS):
            if action == a:
                continue
            self._policy[s, a] = self.eps / N_ACTIONS

        assert np.isclose(self._policy[s, :].sum(), 1.0)


class SarsaLearning(LearningInterface):
    def __init__(self, alpha: float, gamma: float, world: WorldInterface):
        self._q_state_actions = np.zeros((world.n_states, N_ACTIONS))
        self._q_state_actions[world.end_state, :] = 0
        self.a = alpha
        self.g = gamma

    def update_value(self, s: int, s2: int, action: int, action2: int, reward: float):
        self._q_state_actions[s, action] += self.a * (
                    reward + self.g * self._q_state_actions[s2, action2] - self._q_state_actions[s, action])

    def get_best_action(self, s: int) -> int:
        return int(np.argmax(self._q_state_actions[s, :]))


class nStepSarsaLearning(LearningInterface):
    def __init__(self, n: int, alpha: float, gamma: float, world: WorldInterface):
        self.n = n
        self._q_state_actions = np.zeros((world.n_states, N_ACTIONS))
        self._q_state_actions[world.end_state, :] = 0
        self.a = alpha
        self.g = gamma  # discount weight
        self._counter = 0
        self.reward_queue = deque(maxlen=n)
        self.state_queue = deque(maxlen=n)
        self.action_queue = deque(maxlen=n)

    def update_value(self, s: int, s2: int, action: int, action2: int, reward: float):
        self._counter += 1
        # oldest on the left, newest on the right
        self.reward_queue.append(reward)
        self.state_queue.append(s)
        self.action_queue.append(action)
        # can only start updating after the nth step
        if self._counter < self.n:
            return
        self._update_value(s2, action2)

    def _update_value(self, s2: int, action2: int):
        target = 0
        # the restults after more steps have a larger discount factor gamma**i
        n = len(self.reward_queue)
        for i in range(n):
            target += self.g ** i * self.reward_queue[i]
        target += self.g ** n * self._q_state_actions[s2, action2]  # when s2 is terminal, this adds nothing

        # we update the oldest state with the n steps of information postceeding it
        n_state_index = self.state_queue[0]
        n_action_index = self.action_queue[0]

        self._q_state_actions[n_state_index, n_action_index] += self.a * (
                    target - self._q_state_actions[n_state_index, n_action_index])

    def get_best_action(self, s: int) -> int:
        return int(np.argmax(self._q_state_actions[s, :]))

    def end(self, s_end: int, action: int):
        while self.state_queue:
            self.state_queue.popleft()
            self.action_queue.popleft()
            self.reward_queue.popleft()
            self._update_value(s_end, action)

    def reset(self):
        self._counter = 0
        self.reward_queue.clear()
        self.state_queue.clear()
        self.action_queue.clear()


class QLearningLearning(LearningInterface):
    def __init__(self, alpha: float, gamma: float, world: WorldInterface):
        self._q_state_actions = np.zeros((world.n_states, N_ACTIONS))
        self._q_state_actions[world.end_state, :] = 0
        self.a = alpha
        self.g = gamma

    def update_value(self, s: int, s2: int, action: int, action2: int, reward: float):
        q_best_next = np.max(self._q_state_actions[s2, :])
        self._q_state_actions[s, action] += self.a * (reward + self.g * q_best_next - self._q_state_actions[s, action])

    def get_best_action(self, s: int) -> int:
        return int(np.argmax(self._q_state_actions[s, :]))

    def greedy_policy_reward(self, initial_state: int, world: WorldInterface) -> float:
        s = initial_state
        r = 0
        for _ in range(world.width * world.height):
            action = int(np.argmax(self._q_state_actions[s, :]))
            s = world.get_new_state(s, action)
            r += world.get_reward(s)
            if s == world.end_state:
                break
        return r


class DoubleQLearningLearning(LearningInterface):
    """Updating the state action value for one set of values for Q, and then picking the action from 
    another Q (with order selected at random) removes posativity bias"""

    def __init__(self, alpha: float, gamma: float, world: WorldInterface):
        self._q_state_actions_1 = np.zeros((world.width, world.height, N_ACTIONS))
        self._q_state_actions_1[world.end[0], world.end[1], :] = 0

        self._q_state_actions_2 = np.zeros((world.width, world.height, N_ACTIONS))
        self._q_state_actions_2[world.end[0], world.end[1], :] = 0

        self.a = alpha
        self.g = gamma

    def update_value(self, s: int, s2: int, action: int, action2: int, reward: float):
        if np.random.random() < 0.5:
            action_index = np.argmax(self._q_state_actions_1[s2, :])
            q_best_next = self._q_state_actions_2[s2, action_index]
            self._q_state_actions_1[s, action] += self.a * (
                        reward + self.g * q_best_next - self._q_state_actions_1[s, action])
        else:
            action_index = np.argmax(self._q_state_actions_2[s2, :])
            q_best_next = self._q_state_actions_1[s2, action_index]
            self._q_state_actions_2[s, action] += self.a * (
                        reward + self.g * q_best_next - self._q_state_actions_2[s, action])

    def get_best_action(self, s: int) -> int:
        return int(np.argmax(self._q_state_actions_1[s, :] + self._q_state_actions_2[s, :]))


class DynaQLearning(PlannedLearningInterface):
    """Implements the Dyna-Q learning algorithm, which combines direct reinforcement learning 
    with model-based planning to improve sample efficiency"""

    def __init__(self, alpha: float, gamma: float, world: WorldInterface, planner: PlannerOrchestraterInterface):
        """
        Parameters:
                alpha (float): Learning rate controlling how much new information overrides old estimates.
                gamma (float): Discount factor determining the importance of future rewards.
                world (WorldInterface): The environment providing state transitions and rewards.
                planner (PlannerOrchestraterInterface): The planning module used to simulate experience 
                    and update Q-values based on the model.
        """
        self._q_state_actions = np.zeros((world.n_states, N_ACTIONS))
        self._q_state_actions[world.end_state, :] = 0
        self.a = alpha
        self.g = gamma
        self.planner = planner

    def get_error(self, s: int, a: int, s2: int, r: float) -> float:
        """The error used when updating the value"""
        q_best_next = np.max(self._q_state_actions[s2, :])
        return (r + self.g * q_best_next - self._q_state_actions[s, a])

    def update_value(self, s: int, s2: int, action: int, action2: int, reward: float):
        self._q_state_actions[s, action] += self.a * self.get_error(s, action, s2, reward)
        self.planner.update_model(s, action, s2, reward)
        self.planner.plan(self)

    def direct_update(self, state: int, action: int, state2: int, reward):
        self._q_state_actions[state, action] += self.a * self.get_error(state, action, state2, reward)

    def get_best_action(self, s: int) -> int:
        return int(np.argmax(self._q_state_actions[s, :]))

    def greedy_policy_reward(self, initial_state: int, world: WorldInterface) -> float:
        s = initial_state
        r = 0
        for _ in range(world.width * world.height):
            action = int(np.argmax(self._q_state_actions[s, :]))
            s = world.get_new_state(s, action)
            r += world.get_reward(s)
            if s == world.end_state:
                break
        return r


class LearningGateway(LearningInterface):
    def __init__(self, method: METHOD_TYPE, *args, **kargs):
        if method == 'sarsa':
            self.method = SarsaLearning(*args, **kargs)
        elif method == 'q_learning':
            self.method = QLearningLearning(*args, **kargs)
        else:
            raise ValueError(f'Unknown method {method}')

    def update_value(self, s: int, s2: int, action, action2, reward):
        self.method.update_value(s, s2, action, action2, reward)

    def get_best_action(self, s: int) -> int:
        return self.method.get_best_action(s)

    def greedy_policy_reward(self, initial_state, world: WorldInterface) -> float:
        s, y = initial_state
        r = 0
        for _ in range(world.width * world.height):
            action = int(np.argmax(self.method._q_state_actions[s, :]))
            s = world.get_new_state(s, action)
            r += world.get_reward(s)
            if s == world.end_state:
                break
        return r


class WindyWorld(WorldInterface):
    def __init__(self, wind_vec: list[int], start: tuple[int, int], end: tuple[int, int], width: int, height: int):
        self.wind_vec = wind_vec
        self.start = start
        self.end = end
        self.width = width
        self.height = height
        self.encoder = GridStateEncoder2D(width, height)

    def get_reward(self, s: int) -> float:
        return 0 if s == self.end_state else -1

    def get_new_state(self, s: int, action: int) -> int:
        state_x, state_y = self.encoder.decode(s)
        x_dif = 1 if action == ActionIndex.right else -1 if action == ActionIndex.left else 0
        y_dif = 1 if action == ActionIndex.up else -1 if action == ActionIndex.down else 0
        new_state_x = min(max(state_x + x_dif, 0), self.width - 1)
        new_state_y = min(max(state_y + y_dif + self.wind_vec[new_state_x], 0), self.height - 1)
        return self.encoder.encode(new_state_x, new_state_y)


class CliffWorld(WorldInterface):
    def __init__(self, start: tuple[int, int], end: tuple[int, int], width: int, height: int):
        self.start = start
        self.end = end
        self.width = width
        self.height = height
        self.encoder = GridStateEncoder2D(width, height)

    def get_reward(self, s: int) -> float:
        if s == self.end_state:
            return 0
        x, y = self.encoder.decode(s)
        if y == 0 and x != 0 and x != self.width - 1:  # cliff along the bottom
            return - 100
        else:
            return -1

    def get_new_state(self, s: int, action: int) -> int:
        state_x, state_y = self.encoder.decode(s)
        x_dif = 1 if action == ActionIndex.right else -1 if action == ActionIndex.left else 0
        y_dif = 1 if action == ActionIndex.up else -1 if action == ActionIndex.down else 0
        new_state_x = min(max(state_x + x_dif, 0), self.width - 1)
        new_state_y = min(max(state_y + y_dif, 0), self.height - 1)
        return self.encoder.encode(new_state_x, new_state_y)


def find_state_action_value(learning: LearningInterface, world: WorldInterface, policy: PolicyInterface,
                            n_eps: int = 1000):
    """Using the policy and learning method to find the optimal actions from the inital to terminal state in the world across multiple episodes"""
    for i in range(n_eps):
        s = world.start_state
        action = policy.use_policy(s)
        while s != world.end_state:
            s2 = world.get_new_state(s, action)
            action2 = policy.use_policy(s2)
            reward = world.get_reward(s2)
            learning.update_value(s, s2, action, action2, reward)
            best_action = learning.get_best_action(s)
            policy.update_policy(s, best_action)
            s, action = s2, action2
        learning.end(s, action)
        learning.reset()

    world.print_max_prob_path(policy)
    return


if __name__ == "__main__":
    # ------- WIND --------

    _width, _height = 10, 7
    _start = (0, 4)
    _end = (7, 3)
    _wind_vec = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    _world = WindyWorld(_wind_vec, _start, _end, _width, _height)
    _policy = EpsilonGreedyPolicy(0.1, _width * _height)
    _learning = LearningGateway('sarsa', 0.1, 1, _world)
    _sarsa_state_value_reward = find_state_action_value(_learning, _world, _policy)

    _world = WindyWorld(_wind_vec, _start, _end, _width, _height)
    _policy = EpsilonGreedyPolicy(0.1, _width * _height)
    _learning = LearningGateway('q_learning', 0.1, 1, _world)
    _q_learning_state_value_reward = find_state_action_value(_learning, _world, _policy)

    # ------- CLIFF --------

    _width, _height = 12, 4
    _world = CliffWorld((0, 0), (_width - 1, 0), _width, _height)
    _policy = EpsilonGreedyPolicy(0.1, _width * _height)
    _learning = LearningGateway('sarsa', 0.1, 1, _world)
    _sarsa_state_value_reward = find_state_action_value(_learning, _world, _policy)

    _width, _height = 12, 4
    _world = CliffWorld((0, 0), (_width - 1, 0), _width, _height)
    _policy = EpsilonGreedyPolicy(0.1, _width * _height)
    _learning = LearningGateway('q_learning', 0.1, 1, _world)
    _sarsa_state_value_reward = find_state_action_value(_learning, _world, _policy)

    # -------- planner ------ 

    _width, _height = 10, 7
    _start = (0, 4)
    _end = (7, 3)
    _wind_vec = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    # hyper params
    _eps = 0.1
    _k = 0.1
    _theta = 0.1
    _alpha = 0.1
    _gamma = 1
    _logistic_max = 2
    _logistic_growth = 0.1
    _world = WindyWorld(_wind_vec, _start, _end, _width, _height)
    _policy = EpsilonGreedyPolicy(_eps, _width * _height)
    _model = DecayingModel(_world.n_states, N_ACTIONS, _k,
                           lambda x: _logistic_max * np.exp(_logistic_growth * x) / (1 + np.exp(_logistic_growth * x)))
    _planner = PrioritizedSweepingPlanningOrchestrater(_model, _theta)
    _learning = DynaQLearning(_alpha, _gamma, _world, _planner)
    value_reward = find_state_action_value(_learning, _world, _policy)
