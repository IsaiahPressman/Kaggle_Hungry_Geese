from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, row_col
import numpy as np
from typing import *





class Node:
    def __init__(self, parent, position: np.ndarray):
        self.parent = parent
        self.position = position
        # Distance from start node
        self.g = 0
        # Distance to goal node
        self.h = 0

    def __eq__(self, other):
        return np.all(self.position == other.position)

    def __lt__(self, other):
        return self.f < other.f

    @property
    def f(self):
        return self.g + self.h


class Agent:
    def __init__(self, conf):
        conf = Configuration(conf)
        self.board_dims = np.array([conf.rows, conf.columns])
        self.moves = [np.array(a.to_row_col()) for a in Action]

        self.food = []
        self.geese = []
        self.index = None
        self.last_action = None

    def __call__(self, obs, conf):
        obs = Observation(obs)
        conf = Configuration(conf)

        self.food = [self._row_col(f) for f in obs.food]
        self.geese = [[self._row_col(g) for g in goose_locs] for goose_locs in obs.geese]
        self.index = obs.index

        start_nodes = [Node(None, self.wrap(self.geese[self.index][0] + m)) for m in self.moves]
        if self.last_action is not None:
            start_nodes.remove(Node(
                None,
                self.wrap(self.head_position + np.array(self.last_action.opposite().to_row_col()))
            ))
        path_to_food = self.a_star(
            start_nodes,
            [Node(None, f) for f in self.food],
            self.torus_manhattan
        )
        next_position = path_to_food[0]
        for act_idx, move in enumerate(self.moves):
            if np.all(self.wrap(self.head_position + move) == next_position):
                self.last_action = tuple(Action)[act_idx]
                break
        else:
            raise RuntimeError(f'No action found from {self.head_position} to {next_position}')

        return self.last_action.name

    def a_star(self, start_nodes: List[Node], goal_nodes: List[Node], distance_metric):
        # TODO: precompute distance matrix?
        not_visited = sorted([*start_nodes])
        visited = []

        while len(not_visited) > 0:
            current_node = not_visited.pop(0)
            visited.append(current_node)

            if current_node in goal_nodes:
                path = []
                while current_node is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                # Reverse path so that it is in order
                return path[::-1]

            children = [Node(current_node, self.wrap(current_node.position + m)) for m in self.moves]
            for child in children:
                if child not in visited:
                    child.g = current_node.g + 1
                    child.h = min([distance_metric(child.position, goal.position) for goal in goal_nodes])

                    if len([i for i in not_visited if child == i and child.g >= i.g]) == 0:
                        not_visited.append(child)
        # Return None if no path can be found
        return None

    def _row_col(self, position: int) -> np.ndarray:
        return np.array(row_col(position, self.board_dims[1]))

    def wrap(self, position: np.ndarray):
        return (position + self.board_dims) % self.board_dims

    def torus_manhattan(self, a: np.ndarray, b: np.ndarray):
        dists = np.stack([
            np.abs(a - b),
            (a + self.board_dims) - b,
            (b + self.board_dims) - a
        ]).min(axis=0)
        return dists.sum()

    @property
    def head_position(self):
        return self.geese[self.index][0]


agent = None


def call_agent(obs, conf):
    global agent
    if agent is None:
        agent = Agent(conf)

    return agent(obs, conf)
