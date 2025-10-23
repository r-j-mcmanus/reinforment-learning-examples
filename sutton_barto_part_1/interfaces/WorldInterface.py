from pprint import pprint

from part_1.interfaces.EncoderInterface import EncoderInterface
from part_1.interfaces.PolicyInterface import PolicyInterface


class WorldInterface:
    def __init__(self) -> None:
        self.start = (0, 0)
        self.start_state = 0
        self.end = (0, 0)
        self.end_state = 0
        self.width = 0
        self.height = 0
        self.n_states = 0
        self.encoder = EncoderInterface()

    def get_reward(self, s: int) -> float:
        """Return the reward for entering the state"""
        raise NotImplementedError

    def get_new_state(self, s: int, action: int) -> int:
        """Return the new state found given an action in a state"""
        raise NotImplementedError

    def get_grid(self):
        return [['x' for _ in range(self.width)] for __ in range(self.height)]

    def print_max_prob_path(self, policy: PolicyInterface) -> list[list[str]]:
        index_to_action = {
            ActionIndex.left: "l",
            ActionIndex.right: "r",
            ActionIndex.up: "u",
            ActionIndex.down: "d",
        }

        grid = self.get_grid()
        x, y = self.start
        s = self.start_state
        for _ in range(30):
            action = policy.get_most_likely_action(s)
            grid[self.height - y - 1][x] = index_to_action[int(action)]

            s = self.get_new_state(s, action)
            x, y = self.encoder.decode(s)
            if s == self.end_state:
                grid[self.height - y - 1][x] = 'e'
                break

        pprint(grid)
        return grid
