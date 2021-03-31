import itertools

# import threading
from datetime import datetime
from time import time

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation
import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba.types import int32, float32, void, Tuple

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation
from scipy import stats
import sys
import torch
from torch import nn
import torch.nn.functional as F

# See https://www.kaggle.com/c/google-football/discussion/191257
sys.path.append('/kaggle_simulations/agent/')
from hungry_geese.config import *
from hungry_geese.utils import ActionMasking, row_col
from hungry_geese.env import goose_env as ge
from hungry_geese.env.lightweight_env import make_from_state
from hungry_geese.mcts.basic_mcts import BasicMCTS
from hungry_geese import models

# The evaluation episode seems to run with worse resources
# than all following episodes, such that the JIT compilation
# takes more than 60s and times out. Below is a hack that
# avoids compiling a couple functions if the time is before
# a hardcoded timestamp.
EVAL_TIME = datetime.strptime("2021-03-25 15:34:00", "%Y-%m-%d %H:%M:%S")
print("eval time: ", EVAL_TIME)
print("now: ", datetime.now())
if datetime.now() > EVAL_TIME:
    JIT_COMPILE_SIMULATION_FUNCTIONS = True
    print("JIT_COMPILE_SIMULATION_FUNCTIONS = True")
else:
    JIT_COMPILE_SIMULATION_FUNCTIONS = False
    print("JIT_COMPILE_SIMULATION_FUNCTIONS = False")

np.random.seed(0)


SNAKES = 4
SNACKS = 2
MOVES = 4
SNAKE_MAXLEN = 99
EPISODE_STEPS = 200
COLS = 11
ROWS = 7
HUNGER_RATE = 40

MIN_ROLLOUT_DEPTH = 1
BRANCHING_FACTOR = 4
RECURSION = 1
N_SIM_PER_ROUND = 100 if JIT_COMPILE_SIMULATION_FUNCTIONS else 1
EXPLORATION = 1.0
EXPLORATION_2 = 0.5
RISK_AVOIDANCE = 1.0
ALPHA_INERTIA = 1.0
FLOOD_DEPTH = 8
ACCEPTED_THRESHOLD = 0.25
# ^ I set these by replaying bad decisions and changing
# values until the agent reliably made a good decision

JIT_ARGS = {
    "nopython": True,
    # "inline": "never",
    # "parallel": True,
    # "nogil": True,
    "error_model": "numpy",
}

# Moves:
# 0 NORTH
# 1 EAST
# 2 SOUTH
# 3 WEST


OFFSETS = np.array([a.to_row_col() for a in Action], dtype="int32")
OPPOSITES = np.array([a.opposite().value - 1 for a in Action], dtype="int32")
RANK_SCORES = np.linspace(-1, 1, SNAKES, dtype="float32")


@jit(int32(int32, int32, int32, int32), **JIT_ARGS)
def distance(r1, c1, r2, c2):
    diff_r = min((r1 - r2) % ROWS, (r2 - r1) % ROWS)
    diff_c = min((c1 - c2) % COLS, (c2 - c1) % COLS)
    return diff_r + diff_c


def row_col(pos):
    return pos // COLS, pos % COLS


# Indexes:
# s = snake
# b = body segment
# r = row
# c = column
# m = move


state_spec = [
    ("snakes", int32[:, :, :]),  # s, b, rc
    ("snake_array", int32[:, :]),  # r, c
    ("head_ptr", int32[:]),  # s
    ("tail_ptr", int32[:]),  # s
    ("last_move", int32[:]),  # s
    ("length", int32[:]),  # s
    ("alive", int32[:]),  # s
    ("ages", int32[:]),  # s
    ("snack_array", int32[:, :]),  # r, c
    ("illegal_moves", int32[:, :]),  # s, m
    ("step", int32),
    ("done", int32),
    ("index", int32),
]


@jitclass(state_spec)
class State:
    def __init__(self):
        self.snakes = np.zeros((SNAKES, SNAKE_MAXLEN, 2), dtype="int32")
        self.snake_array = np.zeros((ROWS, COLS), dtype="int32")
        self.head_ptr = np.zeros(SNAKES, dtype="int32")
        self.tail_ptr = np.zeros(SNAKES, dtype="int32")
        self.last_move = -np.ones(SNAKES, dtype="int32")
        self.length = np.ones(SNAKES, dtype="int32")
        self.alive = np.ones(SNAKES, dtype="int32")
        self.ages = np.zeros(SNAKES, dtype="int32")
        self.snack_array = np.zeros((ROWS, COLS), dtype="int32")
        self.illegal_moves = np.zeros((SNAKES, MOVES), dtype="int32")
        self.step = 0
        self.done = 0
        self.index = -1

    def head(self, s):
        return self.snakes[s, self.head_ptr[s]]

    def tail(self, s):
        return self.snakes[s, self.tail_ptr[s]]

    @property
    def rewards(self):
        return self.ages * (SNAKE_MAXLEN + 1) + self.length

    def _move_snake(self, s, m):
        offset_r, offset_c = OFFSETS[m]
        old_r, old_c = self.head(s)
        new_r = (old_r + offset_r) % ROWS
        new_c = (old_c + offset_c) % COLS
        self.head_ptr[s] = (self.head_ptr[s] + 1) % SNAKE_MAXLEN
        self.snakes[s, self.head_ptr[s]] = new_r, new_c
        self.snake_array[new_r, new_c] += 1
        tail_r, tail_c = self.tail(s)
        self.snake_array[tail_r, tail_c] -= 1
        self.tail_ptr[s] = (self.tail_ptr[s] + 1) % SNAKE_MAXLEN
        self.last_move[s] = m

    def _grow_snake(self, s):
        if self.length[s] < SNAKE_MAXLEN:
            self.tail_ptr[s] = (self.tail_ptr[s] - 1) % SNAKE_MAXLEN
            tail_r, tail_c = self.tail(s)
            self.snake_array[tail_r, tail_c] += 1

    def _shrink_snake(self, s):
        if self.length[s] == 1:
            self._kill_snake(s)
        else:
            tail_r, tail_c = self.tail(s)
            self.snake_array[tail_r, tail_c] -= 1
            self.tail_ptr[s] = (self.tail_ptr[s] + 1) % SNAKE_MAXLEN

    def _kill_snake(self, s):
        b = self.tail_ptr[s]
        while True:
            r, c = self.snakes[s, b]
            self.snake_array[r, c] -= 1
            if b == self.head_ptr[s]:
                break
            b = (b + 1) % SNAKE_MAXLEN
        self.alive[s] = 0

    def _has_self_collision(self, s):
        head_r, head_c = self.head(s)
        b = self.tail_ptr[s]
        while True:
            if b == self.head_ptr[s]:
                break
            r, c = self.snakes[s, b]
            if r == head_r and c == head_c:
                return True
            b = (b + 1) % SNAKE_MAXLEN
        return False

    def update_illegal_moves(self):
        self.illegal_moves[:] = 0
        for s in range(SNAKES):
            if self.alive[s]:
                if self.last_move[s] >= 0:
                    opp_move = OPPOSITES[self.last_move[s]]
                    self.illegal_moves[s, opp_move] = 1
            else:  # dead snakes forced to 'move' north
                self.illegal_moves[s, 1:] = 1

    def transition(self, moves):

        self.step += 1

        # it seems like in the kaggle implementation, food is
        # eaten by the first snake in the index if there is a
        # conflict, i.e. snake 0 and 1 both move to food, 0
        # gets it (and grows), 1 does not

        # move snakes + immediate effects
        snacks_eaten = 0
        for s in range(SNAKES):
            if self.alive[s]:
                self._move_snake(s, moves[s])
                r, c = self.head(s)
                if self.snack_array[r, c] == 1:
                    snacks_eaten += 1
                    self.snack_array[r, c] = 0
                    self._grow_snake(s)
                if self._has_self_collision(s):
                    self._kill_snake(s)
                if self.alive[s]:
                    if self.step % HUNGER_RATE == 0:
                        self._shrink_snake(s)

        # resolve collisions
        to_kill = np.zeros(SNAKES, dtype="int32")
        for r in range(ROWS):
            for c in range(COLS):
                if self.snake_array[r, c] > 1:
                    for s in range(SNAKES):
                        head_r, head_c = self.head(s)
                        if (r, c) == (head_r, head_c):
                            to_kill[s] = 1
        for s in range(SNAKES):
            if to_kill[s]:
                self._kill_snake(s)

        # update lengths
        for s in range(SNAKES):
            if self.alive[s]:
                self.length[s] = (
                    self.head_ptr[s] - self.tail_ptr[s]
                ) % SNAKE_MAXLEN + 1
                if self.length[s] == 0:
                    return self._kill_snake(s)

        self.update_illegal_moves()
        self.ages += self.alive

        # refresh snacks
        for _ in range(snacks_eaten):
            rs, cs = np.where((self.snake_array + self.snack_array) == 0)
            choice = np.random.choice(rs.size)
            r, c = rs[choice], cs[choice]
            self.snack_array[r, c] = 1

        # are we done?
        self.done = False
        if self.step >= EPISODE_STEPS:
            self.done = True
        elif self.alive.sum() <= 1:
            self.done = True
        elif self.index >= 0:
            if not self.alive[self.index]:
                self.done = True

        return self


state_type = State.class_type.instance_type


display_dict = {
    0: "_",
    1: "a",
    2: "b",
    3: "c",
    4: "d",
    9: "*",
    10: "_",
    11: "A",
    12: "B",
    13: "C",
    14: "D",
}


def display(state):
    print("step:", state.step)
    print("lengths:", state.length * state.alive)
    print("reward", state.rewards)
    print("prop score:", get_prop_scores(state))
    print("flood score:", get_flood_scores(state))
    print()
    array = np.zeros((ROWS, COLS), dtype="int32")
    for s in range(SNAKES):
        if state.alive[s]:
            b = state.tail_ptr[s]
            while True:
                r, c = state.snakes[s, b]
                array[r, c] += 1 * (s + 1)
                if b == state.head_ptr[s]:
                    break
                b = (b + 1) % SNAKE_MAXLEN
            array[r, c] += 10
    array += state.snack_array * 9
    for row in array:
        for cell in row:
            print(display_dict.get(cell, "?"), end=" ")
        print()


def _copy_state(source, target):
    target.snakes[:] = source.snakes[:]
    target.snake_array[:] = source.snake_array[:]
    target.head_ptr[:] = source.head_ptr[:]
    target.tail_ptr[:] = source.tail_ptr[:]
    target.last_move[:] = source.last_move[:]
    target.length[:] = source.length[:]
    target.alive[:] = source.alive[:]
    target.ages[:] = source.ages[:]
    target.snack_array[:] = source.snack_array[:]
    target.illegal_moves[:] = source.illegal_moves[:]
    target.step = source.step
    target.done = source.done
    target.index = source.index
    return target


@jit(int32(int32[:]), **JIT_ARGS)
def moves_to_key(moves):
    key = 0
    for s in range(SNAKES):
        key += MOVES ** s * moves[s]
    return key


key_to_move_array = np.zeros((SNAKES ** MOVES, MOVES), dtype="int32")
for _moves in itertools.product(range(MOVES), repeat=SNAKES):
    key = moves_to_key(np.array(_moves, dtype="int32"))
    key_to_move_array[key, :] = _moves


tree_spec = [
    ("parent", int32[:]),
    ("children", int32[:, :]),
    ("depth", int32[:]),
    ("moves", int32[:, :]),
    ("full_visits", float32[:, :]),
    ("visits", float32[:, :, :]),
    ("values", float32[:, :, :]),
    ("alphas", float32[:, :, :]),
    ("ptr", int32),
    ("size", int32),
    ("max_size", int32),
]


@jitclass(tree_spec)
class Tree:
    def __init__(self, max_size):
        self.parent = -np.empty(max_size, dtype="int32")
        self.children = np.empty((max_size, MOVES ** SNAKES), dtype="int32")
        self.depth = np.empty(max_size, dtype="int32")
        self.moves = -np.empty((max_size, SNAKES), dtype="int32")
        self.full_visits = np.empty((max_size, MOVES ** SNAKES), dtype="float32")
        self.visits = np.empty((max_size, SNAKES, MOVES), dtype="float32")
        self.values = np.empty((max_size, SNAKES, MOVES), dtype="float32")
        self.alphas = np.empty((max_size, SNAKES, MOVES), dtype="float32")
        self.max_size = max_size
        self.init_values()

    def init_values(self):
        self.parent[:] = -1
        self.children[:] = 0
        self.depth[:] = 0
        self.moves[:] = -1
        self.full_visits[:] = 0.0
        self.visits[:] = 0.0
        self.values[:] = 0.0
        self.alphas[:] = RANK_SCORES[-2]
        self.ptr = 0
        self.size = 1

    def get_parent(self):
        parent_ptr = self.parent[self.ptr]
        if parent_ptr >= 0:
            self.ptr = parent_ptr
            return True
        else:
            return False

    def get_child(self, moves):
        key = moves_to_key(moves)
        child_ptr = self.children[self.ptr, key]
        if child_ptr > 0:
            self.ptr = child_ptr
            return True
        else:
            return False

    def make_child(self, moves):
        child_ptr = self.size
        if child_ptr == self.max_size:
            return False
        self.size += 1
        key = moves_to_key(moves)
        self.children[self.ptr, key] = child_ptr
        self.parent[child_ptr] = self.ptr
        self.depth[child_ptr] = self.depth[self.ptr] + 1
        self.moves[child_ptr, :] = moves
        self.ptr = child_ptr
        return True


tree_type = Tree.class_type.instance_type


@jit(float32[:, :](state_type), **JIT_ARGS)
def get_priors(state):
    priors = np.ones((SNAKES, MOVES), dtype="float32")
    n_snakes_alive = state.alive.sum()
    for s in range(SNAKES):
        if state.alive[s]:
            head_r, head_c = state.head(s)
            for m, (diff_r, diff_c) in enumerate(OFFSETS):
                if state.illegal_moves[s, m]:
                    priors[s, m] = 0.0
                    continue
                new_r, new_c = (head_r + diff_r) % ROWS, (head_c + diff_c) % COLS
                if state.snake_array[new_r, new_c]:
                    priors[s, m] = 0.00001
                mult = 1.0
                for s2 in range(SNAKES):
                    if state.alive[s2]:
                        head2_r, head2_c = state.head(s2)
                        tail2_r, tail2_c = state.tail(s2)
                        if new_r == tail2_r and new_c == tail2_c:
                            mult *= 100_000.0  # cancel out the 0.00001
                        if s == state.index:
                            if s != s2:
                                if distance(new_r, new_c, head2_r, head2_c) == 1:
                                    if (
                                        n_snakes_alive > 2
                                        or state.length[s] <= state.length[s2]
                                    ):
                                        mult *= 0.1  # unlikely but not impossible
                priors[s, m] *= mult
        else:
            priors[s, 1:] = 0.0
    return priors


# @jit(float32[:](state_type), **JIT_ARGS)
def get_basic_scores(state):
    scores = np.zeros(SNAKES, dtype="float32")
    ranked = np.argsort(state.rewards)
    RANK_SCORES = np.linspace(-1, 1, SNAKES).astype(np.float32)
    for rank in range(SNAKES):
        s = ranked[rank]
        scores[s] = RANK_SCORES[rank]
    for rank in range(SNAKES):
        s = ranked[rank]
        rew = state.rewards[s]
        tied_idxs = np.where(state.rewards == rew)[0]
        n_tied = len(tied_idxs)
        if n_tied > 1:
            scores[tied_idxs] = scores[tied_idxs].mean()
    return scores


@jit(Tuple((float32[:], float32))(state_type), **JIT_ARGS)
def get_min_and_excess_scores(state):
    scores = np.zeros(SNAKES, dtype="float32")
    ranked = np.argsort(state.rewards)
    n_alive = state.alive.sum()
    alive_min_score = RANK_SCORES[SNAKES - n_alive]
    for rank in range(SNAKES):
        s = ranked[rank]
        if state.alive[s]:
            scores[s] = alive_min_score
        else:
            scores[s] = RANK_SCORES[rank]
    if n_alive <= SNAKES - 2:  # at least two are dead
        for rank in range(SNAKES):
            s = ranked[rank]
            if not state.alive[s]:
                rew = state.rewards[s]
                tied_idxs = np.where(state.rewards == rew)[0]
                n_tied = len(tied_idxs)
                if n_tied > 1:
                    scores[tied_idxs] = scores[tied_idxs].mean()
    excess_score = -scores.sum()
    return scores, excess_score


# @jit(float32[:](state_type), **JIT_ARGS)
def get_prop_scores(state):
    scores, excess_score = get_min_and_excess_scores(state)
    if excess_score > 0:
        lengths = state.length * state.alive
        props = lengths / lengths.sum()
        scores += props * excess_score
    return scores


@jit(float32[:, :](state_type), **JIT_ARGS)
def get_flood_data(state):
    QUEUE_SIZE = ROWS * COLS * SNAKES  # upper limit for simultaneous changes?
    queue = np.zeros((QUEUE_SIZE, 3), dtype="int32")  # s, r, c
    read_ptr, write_ptr = 0, 0
    obstacles = np.zeros((ROWS, COLS), dtype="int32")
    flood = -np.ones((ROWS, COLS), dtype="int32")
    owned = np.zeros((SNAKES, ROWS, COLS), dtype="int32")
    flood_data = np.zeros((SNAKES, 3), dtype="float32")
    n_owned = flood_data[:, 0]
    n_snacks = flood_data[:, 1]
    max_flood = flood_data[:, 2]
    for s in range(SNAKES):
        s = int32(s)
        if state.alive[s]:
            b = state.tail_ptr[s]
            val = 1
            while True:
                r, c = state.snakes[s, b]
                obstacles[r, c] = val
                if b == state.head_ptr[s]:
                    flood[r, c] = 0
                    queue[write_ptr] = s, r, c
                    write_ptr = (write_ptr + 1) % QUEUE_SIZE
                    break
                val += 1
                b += 1
    for d in range(1, FLOOD_DEPTH):
        frozen_write_ptr = write_ptr
        while True:
            s, r, c = queue[read_ptr]
            if s >= 0:  # can be set to -1 below in case of conflicts
                for _, (diff_r, diff_c) in enumerate(OFFSETS):
                    _r = int32((r + diff_r) % ROWS)  # TODO: conversion?
                    _c = int32((c + diff_c) % COLS)  # TODO: conversion?
                    if obstacles[_r, _c] - d < 1:
                        if flood[_r, _c] == -1:
                            if owned[s, _r, _c] == 0:
                                owned[s, _r, _c] = 1
                                queue[write_ptr] = s, _r, _c
                                write_ptr = (write_ptr + 1) % QUEUE_SIZE
            read_ptr = (read_ptr + 1) % QUEUE_SIZE
            if read_ptr == frozen_write_ptr:
                break
        if write_ptr != read_ptr:
            ptr = read_ptr
            while True:
                s, r, c = queue[ptr]
                flood[r, c] = d
                n_snakes_in_cell = owned[:, r, c].sum()
                val = 1 / n_snakes_in_cell
                n_owned[s] += val
                n_snacks[s] += val * state.snack_array[r, c]
                max_flood[s] = d
                if n_snakes_in_cell > 1:
                    queue[ptr] = -1  # if conflict then this is not a node
                ptr = (ptr + 1) % QUEUE_SIZE
                if ptr == write_ptr:
                    break
    return flood_data


FLOOD_BETA = np.array(
    [
        # [0.3025],  # length
        # [-0.0326],  # n_owned
        # [0.0169],  # n_snacks
        # [0.117],  # max_flood
        [0.6385],  # length
        [0.0694],  # n_owned
        [0.0257],  # n_snacks
        [0.0319],  # max_flood
        # [0.82],  #  length
        # [0.11],  # n_owned
        # [0.04],  # n_snacks
        # [0.03],  # max_flood
    ],
    dtype="float32",
)


@jit(float32[:](state_type), **JIT_ARGS)
def get_flood_scores(state):
    flood_data = get_flood_data(state)
    flood_data /= (flood_data.sum(axis=0) + 0.0000001).reshape((1, -1))
    lengths = state.length * state.alive
    props = lengths / (lengths.sum() + 0.0000001)
    X = np.empty((SNAKES, 4), dtype="float32")
    X[:, 0] = props
    X[:, 1:4] = flood_data
    pred = np.dot(X, FLOOD_BETA).ravel()
    pred /= pred.sum() + 0.0000001
    min_scores, excess_score = get_min_and_excess_scores(state)
    return min_scores + excess_score * pred


def _explore_moves(state, tree, moves):
    Q = tree.values[tree.ptr] / (tree.visits[tree.ptr] + 1.0)
    U = tree.visits[tree.ptr].sum(axis=1) ** 0.5 / (tree.visits[tree.ptr] + 1.0)
    P = get_priors(state)
    for s in range(SNAKES):
        if s != state.index:
            continue
        for m in range(MOVES):
            # this below is a bit hacky, but helps alphas work much
            # better and avoid mostly useless rollouts:
            if P[s, m] < 0.001:
                curr_alpha = tree.alphas[tree.ptr][s, m]
                N = ALPHA_INERTIA + tree.visits[tree.ptr][s, m]
                # N = 0.0
                new_alpha = curr_alpha * (N / (N + 1.0)) + -1.0 * (1.0 / (N + 1.0))
                tree.alphas[tree.ptr, s, m] = new_alpha
    for s in range(SNAKES):
        P[s] /= P[s].sum()
    X = Q + EXPLORATION * P * U + RISK_AVOIDANCE * tree.alphas[tree.ptr]
    X2 = (
        (tree.full_visits[tree.ptr].sum() + 1.0) ** 0.7
        / (tree.full_visits[tree.ptr] + 1.0)
        * EXPLORATION_2
    )
    for _key, _moves in enumerate(key_to_move_array):
        for s in range(SNAKES):
            X2[_key] *= P[s, _moves[s]]
        for s in range(SNAKES):
            X2[_key] += X[s, _moves[s]]
    key = np.argmax(X2)
    moves[:] = key_to_move_array[key]
    return moves


def choose_moves(state, tree, moves):
    for s in range(SNAKES):
        moves[s] = np.argmax(tree.visits[tree.ptr][s])
    return moves


@jit(void(state_type, tree_type), **JIT_ARGS)
def backup(state, tree):
    # scores = get_prop_scores(state)
    scores = get_flood_scores(state)
    at_bottom = True
    while True:
        parent_ptr = tree.parent[tree.ptr]
        if parent_ptr < 0:
            break
        key = moves_to_key(tree.moves[tree.ptr])
        tree.full_visits[parent_ptr, key] += 1.0
        for s in range(SNAKES):
            m = tree.moves[tree.ptr][s]
            tree.visits[parent_ptr, s, m] += 1.0
            tree.values[parent_ptr, s, m] += scores[s]
            if s == state.index:  # opps don't use alpha
                if at_bottom and state.done:
                    curr_alphas = tree.alphas[tree.ptr][s]
                    N = ALPHA_INERTIA + tree.visits[tree.ptr][s].sum()
                    # N = 0.0
                    new_alphas = curr_alphas * (N / (N + 1.0)) + scores[s] * (
                        1.0 / (N + 1.0)
                    )
                    tree.alphas[tree.ptr, s, :] = new_alphas
                max_alpha = tree.alphas[tree.ptr, s].max()
                curr_alpha = tree.alphas[parent_ptr, s, m]
                if max_alpha < curr_alpha:
                    tree.alphas[parent_ptr, s, m] = max_alpha
        tree.get_parent()
        at_bottom = False


def _simulate(state, tree, moves, recursion):
    min_depth = tree.depth[tree.ptr] + MIN_ROLLOUT_DEPTH
    cached_state = copy_state(state, State())
    cached_ptr = tree.ptr
    for _ in range(BRANCHING_FACTOR):
        while True:
            moves = explore_moves(state, tree, moves)
            has_child = tree.get_child(moves)
            state.transition(moves)
            if not has_child:
                success = tree.make_child(moves)
                if not success:  #  i.e. tree is full
                    tree.ptr = 0
                    return False
            if state.done or (not has_child and tree.depth[tree.ptr] >= min_depth):
                tmp_ptr = tree.ptr
                backup(state, tree)
                break
        if not state.done and recursion:
            tree.ptr = tmp_ptr
            simulate(state, tree, moves, recursion - 1)
        copy_state(cached_state, state)
        tree.ptr = cached_ptr
    return True


if JIT_COMPILE_SIMULATION_FUNCTIONS:
    copy_state = jit(state_type(state_type, state_type), **JIT_ARGS)(_copy_state)
    explore_moves = jit(int32[:](state_type, tree_type, int32[:]), **JIT_ARGS)(_explore_moves)
    simulate = jit(_simulate)
else:
    copy_state = _copy_state
    explore_moves = _explore_moves
    simulate = _simulate


def _simulate_n_rounds(state, tree, n_rounds):
    moves = np.zeros(SNAKES, dtype="int32")
    for _ in range(n_rounds):
        success = simulate(state, tree, moves, RECURSION)
    return success


# if JIT_COMPILE_SIMULATION_FUNCTIONS:
if False:
    simulate_n_rounds = jit(_simulate_n_rounds)
else:
    simulate_n_rounds = _simulate_n_rounds


def infer_move(diff_r, diff_c):
    if diff_r < 0:
        if diff_r > -2:
            return 0
        else:
            return 2
    elif diff_r > 0:
        if diff_r < 2:
            return 2
        else:
            return 0
    elif diff_c > 0:
        if diff_c < 2:
            return 1
        else:
            return 3
    elif diff_c < 0:
        if diff_c > -2:
            return 3
        else:
            return 1
    else:
        return -1


MOVE_STRINGS = [a.name for a in Action]


BOARD_DIMS = np.array([N_ROWS, N_COLS])


def wrap(position: np.ndarray) -> np.ndarray:
    assert position.shape == (2,), f'{position.shape}'
    return (position + BOARD_DIMS) % BOARD_DIMS


# Precompute directions_dict for get_direction function
DIRECTIONS_DICT = {tuple(wrap(np.array(act.to_row_col()))): act for act in Action}


def get_direction(from_loc: np.ndarray, to_loc: np.ndarray) -> Action:
    return DIRECTIONS_DICT[tuple(wrap(to_loc - from_loc))]


def action_mask_func(state):
    return ActionMasking.LETHAL.get_action_mask(state)


def terminal_value_func(state):
    agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
    ranks_rescaled = 2. * agent_rankings / (len(state) - 1.) - 1.
    return ranks_rescaled


class Agent:
    def __init__(
        self,
        obs,
        conf,
        sim_time=0.85,
        tree_size=100_000,
        manage_overage=True,
    ):
        self.sim_time = sim_time
        self.tree_size = tree_size
        self.manage_overage = manage_overage
        self.overage_used = 0
        self.old_overage = None
        self.state = None
        self.tree = None

        self.index = obs.index

        obs_type = ge.ObsType.COMBINED_GRADIENT_OBS
        n_channels = 64
        activation = nn.ReLU
        model_kwargs = dict(
            block_class=models.BasicConvolutionalBlock,
            conv_block_kwargs=[
                dict(
                    in_channels=obs_type.get_obs_spec()[-3],
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
            ],
            squeeze_excitation=True,
            cross_normalize_value=True,
        )
        self.model = models.FullConvActorCriticNetwork(**model_kwargs)
        self.model.load_state_dict(torch.load('/kaggle_simulations/agent/cp.pt'))
        self.model.eval()

        self.obs_type = obs_type
        self.nn_pred_processor = BasicMCTS(
            action_mask_func=action_mask_func,
            actor_critic_func=self.actor_critic_func,
            terminal_value_func=terminal_value_func,
            c_puct=1.,
            include_food=False,
        )
        self.last_head_locs = [row_col(goose[0]) for goose in obs.geese]
        self.last_actions = [Action.NORTH for _ in range(4)]

    def preprocess(self, obs: Observation, conf: Configuration):
        for goose_idx, goose in enumerate(obs.geese):
            if len(goose) > 0:
                if self.last_head_locs[goose_idx] is not None and obs.step > 0:
                    self.last_actions[goose_idx] = get_direction(np.array(self.last_head_locs[goose_idx]),
                                                                 np.array(row_col(goose[0])))
                else:
                    self.last_actions[goose_idx] = Action.NORTH
                self.last_head_locs[goose_idx] = row_col(goose[0])
            else:
                self.last_actions[goose_idx] = Action.NORTH
                self.last_head_locs[goose_idx] = None

    def __call__(self, obs, config):

        start_time = time()

        self._load_observation(obs)
        self.nn_pred_processor.reset()
        self.preprocess(obs, config)
        self._reset_tree()

        if self.manage_overage and self.state.step >= 2:
            if self.overage_used < 0.00001 and self.old_overage > 2:
                self.sim_time += 0.005
            elif self.overage_used > 0 and self.old_overage < 30:
                self.sim_time -= 0.01
            elif self.overage_used > 0.5:
                self.sim_time -= 0.1

        setup_time = time() - start_time

        # Get NN prediction (nn_pred_processor performs action masking)
        env = make_from_state(obs, self.last_actions)
        root_node = self.nn_pred_processor.run_mcts(
            env=env,
            n_iter=1,
        )
        nn_policy = root_node.initial_policies[self.index]
        nn_time = time() - (start_time + setup_time)

        n_sim = 0
        elapsed_time = 0
        while elapsed_time < self.sim_time:
            n_sim += N_SIM_PER_ROUND
            success = simulate_n_rounds(self.state, self.tree, N_SIM_PER_ROUND)
            elapsed_time = time() - start_time
            if not success:
                break

        if self.tree_size - self.tree.size < 2_000:
            self.tree_size += 2_000

        search_policy = (self.tree.visits[0, self.state.index] / self.tree.visits[0, self.state.index].sum())
        eps = 1e-6
        final_policy = np.where(
            search_policy >= ACCEPTED_THRESHOLD,
            nn_policy + eps,
            0.
        )
        final_policy = final_policy / final_policy.sum()

        print(
            f"s/a/l: {self.state.step}/{self.state.index}/{self.state.length[self.state.index]} - "
            f"sim/nod/vis/dep: {n_sim}/{self.tree.size}/{self.tree.visits[self.tree.ptr][0].sum()}/{self.tree.depth.max()} - "
            f"search_pi: {(self.tree.visits[0, self.state.index] / self.tree.visits[0, self.state.index].sum()).round(2)} - "
            f"nn_pi: {nn_policy.round(2)} - "
            f"final_pi: {final_policy.round(2)} - "
            f"Search/nn agree: {search_policy.argmax() == nn_policy.argmax()} - "
            f"se/nn/el/ov: {setup_time:.3f}/{nn_time:.3f}/{elapsed_time:.3f}/{obs['remainingOverageTime']:.3f}"
        )

        move_string = MOVE_STRINGS[final_policy.argmax()]
        return move_string

    def _load_observation(self, obs):

        state = State()

        # copy each goose -> snake
        for s in range(SNAKES):

            goose = obs["geese"][s]

            if not len(goose):
                state.alive[s] = 0
                state.last_move[s] = 0  # may not be true, but should be ok
                if self.state is not None:
                    state.length[s] = self.state.length[s]

            else:
                state.alive[s] = 1
                state.length[s] = len(goose)
                state.head_ptr[s] = len(goose) - 1  # tail_ptr ok at zero

                for b, pos in enumerate(goose[::-1]):
                    r, c = row_col(pos)
                    state.snakes[s, b] = r, c
                    state.snake_array[r, c] += 1

                # infer last move from previous head position
                if self.state is not None:
                    curr_r, curr_c = state.head(s)
                    old_r, old_c = self.state.head(s)
                    state.last_move[s] = infer_move(curr_r - old_r, curr_c - old_c)
                else:
                    state.last_move[s] = -1

        # food/snacks
        for pos in obs["food"]:
            r, c = row_col(pos)
            state.snack_array[r, c] = 1

        # overage
        if self.old_overage is not None:
            self.overage_used = self.old_overage - obs["remainingOverageTime"]
        self.old_overage = obs["remainingOverageTime"]

        # misc
        state.step = obs["step"]
        state.index = obs["index"]
        if self.state is not None:
            state.ages[:] = self.state.ages + state.alive

        state.update_illegal_moves()
        self.state = state

    def _is_tree_empty(self):
        return self.tree.parent[1] < 0

    def _reset_tree(self):
        if self.tree is None:
            self.tree = Tree(self.tree_size)
        elif self.tree_size > self.tree.max_size:
            self.tree = Tree(self.tree_size)
        elif not self._is_tree_empty():
            self.tree.init_values()

    def actor_critic_func(self, state):
        geese = state[0]['observation']['geese']
        n_geese = len(geese)

        obs = ge.create_obs_tensor(state, self.obs_type)
        head_locs = [goose[0] if len(goose) > 0 else -1 for goose in geese]
        still_alive = [agent['status'] == 'ACTIVE' for agent in state]
        with torch.no_grad():
            logits, values = self.model(torch.from_numpy(obs),
                                        torch.tensor(head_locs).unsqueeze(0),
                                        torch.tensor(still_alive).unsqueeze(0))

        # Score the dead geese
        dead_geese_mask = np.array([len(goose) for goose in geese]) == 0
        agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        probs = F.softmax(logits, -1).squeeze(0).numpy().astype(np.float)
        final_values = np.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values.squeeze(0).numpy()
        )

        # Logits should be of shape (4, 4)
        # Values should be of shape (4, 1)
        return probs, final_values[:, np.newaxis]


AGENT = None


def call_agent(obs, conf):
    global AGENT

    obs = Observation(obs)
    conf = Configuration(conf)
    if AGENT is None:
        AGENT = Agent(obs, conf)

    return AGENT(obs, conf)
