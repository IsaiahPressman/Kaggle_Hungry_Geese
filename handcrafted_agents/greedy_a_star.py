import copy
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, row_col
import numpy as np
import random
from scipy.stats import rankdata
from typing import *





class Node:
    def __init__(self, parent, position: tuple, last_action: Action, g: int, h: int):
        self.parent = parent
        self.position = position
        self.last_action = last_action
        # Distance from start node
        self.g = g
        # Distance to goal node
        self.h = h

    def __eq__(self, other):
        return self.position == other.position and self.last_action == other.last_action

    def __lt__(self, other):
        return self.f < other.f
        
    @property
    def f(self):
        return self.g + self.h
    
    @property
    def parent_positions(self):
        if self.parent is None:
            return [self.position]
        else:
            return self.parent.parent_positions + [self.position]


class Agent:
    def __init__(self, conf: Configuration):
        self.board_dims = np.array([conf.rows, conf.columns])
        self.moves = {a: np.array(a.to_row_col()) for a in Action}
        # Precompute distance matrix on startup
        self.distance_matrix = {}
        for x in range(conf.rows):
            for y in range(conf.columns):
                key = (x, y)
                self.distance_matrix[key] = {}
                for x_prime in range(conf.rows):
                    for y_prime in range(conf.columns):
                        key_prime = (x_prime, y_prime)
                        self.distance_matrix[key][key_prime] = self.torus_manhattan(np.array(key), np.array(key_prime))
        # Precompute directions_dict for get_direction function
        self.directions_dict = {tuple(self.wrap(np.array(act.to_row_col()))): act for act in Action}
        
        self.food = []
        self.geese = []
        self.index = None
        self.last_head_locs = [None for i in range(4)]
        self.last_actions = [None for i in range(4)]
        self.occupied = np.zeros((conf.rows, conf.columns, conf.episode_steps))

    def __call__(self, obs: Observation, conf: Configuration):
        self.preprocess(obs, conf)
        my_next_position = self.compute_next_position(obs, conf)
        return self.get_direction(np.array(self.head_position), np.array(my_next_position)).name
    
    def compute_next_position(self, obs: Observation, conf: Configuration):
        remaining_food = copy.copy(self.food)
        found_path = [len(goose) == 0 for goose in self.geese]
        occupied_tracker = self.occupied.copy()
        while len(remaining_food) > 0:
            n_food_remaining = len(remaining_food)
            paths = [[None] * n_food_remaining for i in range(len(self.geese))]
            path_lens = [[float('inf')] * n_food_remaining for i in range(len(self.geese))]
            for goose_idx, goose in enumerate(self.geese):
                if len(goose) > 0 and not found_path[goose_idx]:
                    for food_idx, f in enumerate(remaining_food):
                        path, success = self.a_star(
                            Node(None, goose[0], self.last_actions[goose_idx], 0, self.distance_matrix[goose[0]][f]),
                            [f],
                            occupied_tracker
                        )
                        paths[goose_idx][food_idx] = path
                        if success:
                            path_lens[goose_idx][food_idx] = len(path)
                
            shortest_path_len = np.array(path_lens).min()
            if shortest_path_len == float('inf'):
                break
            is_shortest_path = np.array(path_lens) == shortest_path_len
            if is_shortest_path.any(axis=1).sum() == 1:
                priority_goose_indices = [np.argmax(is_shortest_path.any(axis=1)).item()]
            else:
                tied_geese_indices = np.arange(len(self.geese))[is_shortest_path.any(axis=1)]
                priority_goose_indices = tied_geese_indices[np.array(self.would_collide(tied_geese_indices))]
            for goose_idx in priority_goose_indices:
                # TODO: Logic to avoid running into dead ends
                found_path[goose_idx] = True
                path_idx = np.argmax(is_shortest_path[goose_idx])
                selected_path = paths[goose_idx][path_idx]
                if goose_idx == self.index:
                    return selected_path[1]
                for i, location in enumerate(selected_path):
                    time_in_square = len(self.geese[goose_idx])
                    if (obs.step % conf.hunger_rate) + i + time_in_square > conf.hunger_rate:
                        time_in_square = max(time_in_square - 1, 1)
                    if i + time_in_square >= len(selected_path):
                        time_in_square += 1
                    occupied_tracker[location[0], location[1], i:(i+time_in_square)] = 1
                # If the goose is longer than the path, make sure to account for it's tail not moving when it eats in the occupied matrix
                if len(selected_path) - 1 < len(self.geese[goose_idx]):
                    for i, segment in list(enumerate(self.geese[goose_idx]))[:(len(self.geese[goose_idx]) - len(selected_path) + 2)]:
                        time_in_square = len(goose) - i + 1
                        if (obs.step % conf.hunger_rate) + time_in_square > conf.hunger_rate:
                            time_in_square = max(time_in_square-1, 1)
                        occupied_tracker[segment[0], segment[1], :time_in_square] = 1
                if selected_path[-1] in remaining_food:
                    remaining_food.remove(selected_path[-1])
        
        # TODO: Better logic for when I cannot get to food before others
        if not found_path[self.index]:
            food_dist = min([self.distance_matrix[self.head_position][f] for f in self.food])
            path, success = self.a_star(
                Node(None, self.head_position, self.last_actions[self.index], 0, food_dist),
                self.food,
                occupied_tracker
            )
            # In case no path can be found, reset assumptions about collision preferences
            if len(path) == 1:
                print('No non-colliding path found')
                food_dist = min([self.distance_matrix[self.head_position][f] for f in self.food])
                path, success = self.a_star(
                    Node(None, self.head_position, self.last_actions[self.index], 0, food_dist),
                    self.food,
                    self.occupied
                )
                # If still no path can be found, just continue straight
                if len(path) == 1:
                    print('No path found')
                    return tuple(self.wrap(np.array(self.head_position) +
                                           np.array(self.last_actions[self.index].to_row_col())))
            return path[1]

    def preprocess(self, obs: Observation, conf: Configuration):
        self.food = [self._row_col(f) for f in obs.food]
        self.geese = [[self._row_col(g) for g in goose_locs] for goose_locs in obs.geese]
        self.index = obs.index
        self.occupied = np.zeros((conf.rows, conf.columns, conf.episode_steps-obs.step))
        
        for goose_idx, goose in enumerate(self.geese):
            if len(goose) > 0:
                if self.last_head_locs[goose_idx] is not None:
                    self.last_actions[goose_idx] = self.get_direction(np.array(self.last_head_locs[goose_idx]), np.array(goose[0]))
                self.last_head_locs[goose_idx] = goose[0]
                for i, segment in enumerate(goose):
                    time_in_square = len(goose)-i
                    if (obs.step % conf.hunger_rate) + time_in_square > conf.hunger_rate:
                        time_in_square = max(time_in_square-1, 1)
                    self.occupied[segment[0], segment[1], :time_in_square] = 1
            else:
                self.last_actions[goose_idx] = None
                self.last_head_locs[goose_idx] = None
        
    def a_star(self, start_node: Node, goal_locs: List[tuple], occupied_matrix) -> Tuple[List[Tuple[int, int]], bool]:
        not_visited = [start_node]
        visited = []

        while len(not_visited) > 0:
            current_node = not_visited.pop(0)
            visited.append(current_node)

            if current_node.position in goal_locs:
                return current_node.parent_positions, True
            
            children = []
            for a, m in self.moves.items():
                if current_node.last_action is None or a != current_node.last_action.opposite():
                    new_loc = tuple(self.wrap(np.array(current_node.position) + m))
                    if occupied_matrix[new_loc[0], new_loc[1], current_node.g + 1] == 0 and new_loc not in current_node.parent_positions:
                        goal_dist = min([self.distance_matrix[new_loc][goal] for goal in goal_locs])
                        children.append(Node(current_node, new_loc, a, current_node.g + 1, goal_dist))
            not_visited.extend(children)
        paths = [v.parent_positions for v in visited]
        return paths[np.argmax([len(p) for p in paths])], False

    def would_collide(self, goose_indices: Sequence[int]) -> List[bool]:
        # If 2 geese would remain after the collision, the shorter goose would like to collide
        # In case of ties in this case, one of the geese randomly opts for collision
        # If 0 or 1 geese would remain after the collision, the longer goose would like to collide
        # In case of ties in this case, all geese opt for collision
        current_num_living_geese = np.sum(np.array([len(g) for g in self.geese]) > 0)
        num_geese_after_collision = current_num_living_geese - len(goose_indices)
        goose_lengths = np.array([len(g) for g in self.geese])[np.array(goose_indices)]
        if num_geese_after_collision >= 2:
            assert len(goose_indices) == 2
            if goose_lengths[0] == goose_lengths[1]:
                if random.random() < 0.5:
                    return [True, False]
                else:
                    return [False, True]
            elif goose_lengths[0] < goose_lengths[1]:
                return [True, False]
            else:
                return [False, True]
        else:
            would_collide = []
            for goose_length in goose_lengths:
                would_collide.append(goose_length == max(goose_lengths))
            return would_collide
        
    def get_direction(self, from_loc: np.ndarray, to_loc: np.ndarray) -> Action:
        return self.directions_dict[tuple(self.wrap(to_loc - from_loc))]
    
    def _row_col(self, position: int) -> Tuple[int, int]:
        return row_col(position, self.board_dims[1])

    def wrap(self, position: np.ndarray):
        assert position.shape == (2,)
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
    
    obs = Observation(obs)
    conf = Configuration(conf)
    if agent is None:
        agent = Agent(conf)

    return agent(obs, conf)
