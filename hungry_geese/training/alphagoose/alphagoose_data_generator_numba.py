import itertools
from time import time

import multiprocessing as mp
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from numba import jit, prange, set_num_threads, threading_layer, typed
from numba.experimental import jitclass
from numba.types import float32, int32
from pathlib import Path
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from ...config import *
from ...env import goose_env as ge
from ... import models
from .alphagoose_data_generator import reload_model_weights, save_episodes_worker


NUMBA_THREADS = 8  # doesn't seem to get much faster after 4?
JIT_ARGS = {
    "nopython": True,
    "error_model": "numpy",
}
set_num_threads(NUMBA_THREADS)


# Basic gameplay parameters:
SNACKS = 2
MOVES = 4


# MCTS parameters:
EXPLORATION = 1.0  # how much to prefer un/under-explored nodes


# Self-play data generation parameters:
ENVS = 64  # number of parallel self play environments
NODES_PER_SEARCH_STEP = 8  # number of nodes each env evaluates at once
SELFPLAY_SEARCH_ROUNDS = 25  # how many times to search each step
BUFFER_SIZE = 262144  # this will hold ~1.3k episodes of data
INFO_SHAPE = (N_PLAYERS, 5)
NN_OBS_SHAPE = (N_PLAYERS * 2 + 3, N_ROWS, N_COLS)

# Moves:
# 0 NORTH
# 1 EAST
# 2 SOUTH
# 3 WEST

OFFSETS = np.array([list(a.to_row_col()) for a in Action], dtype="int32")
OPPOSITES = np.array([a.opposite().value - 1 for a in Action], dtype="int32")
RANK_SCORES = np.linspace(-1, 1, N_PLAYERS, dtype="float32")


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
]


@jitclass(state_spec)
class State:
    def __init__(self):
        self.snakes = np.zeros((N_PLAYERS, GOOSE_MAX_LEN, 2), dtype="int32")
        self.snake_array = np.zeros((N_ROWS, N_COLS), dtype="int32")
        self.head_ptr = np.zeros(N_PLAYERS, dtype="int32")
        self.tail_ptr = np.zeros(N_PLAYERS, dtype="int32")
        self.last_move = -np.ones(N_PLAYERS, dtype="int32")
        self.length = np.ones(N_PLAYERS, dtype="int32")
        self.alive = np.ones(N_PLAYERS, dtype="int32")
        self.ages = np.zeros(N_PLAYERS, dtype="int32")
        self.snack_array = np.zeros((N_ROWS, N_COLS), dtype="int32")
        self.illegal_moves = np.zeros((N_PLAYERS, MOVES), dtype="int32")
        self.step = 0
        self.done = 0

    def head(self, s):
        return self.snakes[s, self.head_ptr[s]]

    def tail(self, s):
        return self.snakes[s, self.tail_ptr[s]]

    @property
    def rewards(self):
        return self.ages * (GOOSE_MAX_LEN + 1) + self.length

    def _move_snake(self, s, m):
        offset_r, offset_c = OFFSETS[m]
        old_r, old_c = self.head(s)
        new_r = (old_r + offset_r) % N_ROWS
        new_c = (old_c + offset_c) % N_COLS
        self.head_ptr[s] = (self.head_ptr[s] + 1) % GOOSE_MAX_LEN
        self.snakes[s, self.head_ptr[s]] = new_r, new_c
        self.snake_array[new_r, new_c] += 1
        tail_r, tail_c = self.tail(s)
        self.snake_array[tail_r, tail_c] -= 1
        self.tail_ptr[s] = (self.tail_ptr[s] + 1) % GOOSE_MAX_LEN
        self.last_move[s] = m

    def _grow_snake(self, s):
        if self.length[s] < GOOSE_MAX_LEN:
            self.tail_ptr[s] = (self.tail_ptr[s] - 1) % GOOSE_MAX_LEN
            tail_r, tail_c = self.tail(s)
            self.snake_array[tail_r, tail_c] += 1

    def _shrink_snake(self, s):
        if self.length[s] == 1:
            self._kill_snake(s)
        else:
            tail_r, tail_c = self.tail(s)
            self.snake_array[tail_r, tail_c] -= 1
            self.tail_ptr[s] = (self.tail_ptr[s] + 1) % GOOSE_MAX_LEN

    def _kill_snake(self, s):
        b = self.tail_ptr[s]
        while True:
            r, c = self.snakes[s, b]
            self.snake_array[r, c] -= 1
            if b == self.head_ptr[s]:
                break
            b = (b + 1) % GOOSE_MAX_LEN
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
            b = (b + 1) % GOOSE_MAX_LEN
        return False

    def update_illegal_moves(self):
        self.illegal_moves[:] = 0
        for s in range(N_PLAYERS):
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
        # gets it (and grows), 1 does not, then both die

        # move snakes + immediate effects
        snacks_eaten = 0
        for s in range(N_PLAYERS):
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
        to_kill = np.zeros(N_PLAYERS, dtype="int32")
        for r in range(N_ROWS):
            for c in range(N_COLS):
                if self.snake_array[r, c] > 1:
                    for s in range(N_PLAYERS):
                        head_r, head_c = self.head(s)
                        if (r, c) == (head_r, head_c):
                            to_kill[s] = 1
        for s in range(N_PLAYERS):
            if to_kill[s]:
                self._kill_snake(s)

        # update lengths
        for s in range(N_PLAYERS):
            if self.alive[s]:
                self.length[s] = (
                    self.head_ptr[s] - self.tail_ptr[s]
                ) % GOOSE_MAX_LEN + 1
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
        if self.step >= MAX_NUM_STEPS:
            self.done = True
        elif self.alive.sum() <= 1:
            self.done = True

        return self

    def reset(self):
        self.__init__()
        for s in range(N_PLAYERS):
            rs, cs = np.where(self.snake_array == 0)
            choice = np.random.choice(rs.size)
            r, c = rs[choice], cs[choice]
            self.snake_array[r, c] = 1
            self.snakes[s, 0] = r, c
        self.length[:] = 1
        for _ in range(SNACKS):
            rs, cs = np.where((self.snake_array + self.snack_array) == 0)
            choice = np.random.choice(rs.size)
            r, c = rs[choice], cs[choice]
            self.snack_array[r, c] = 1


_display_dict = {
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
    print("prop score:", get_prop_scores(get_info_from_state(state)))
    print()
    array = np.zeros((N_ROWS, N_COLS), dtype="int32")
    for s in range(N_PLAYERS):
        if state.alive[s]:
            b = state.tail_ptr[s]
            while True:
                r, c = state.snakes[s, b]
                array[r, c] += 1 * (s + 1)
                if b == state.head_ptr[s]:
                    break
                b = (b + 1) % GOOSE_MAX_LEN
            array[r, c] += 10
    array += state.snack_array * 9
    for row in array:
        for cell in row:
            print(_display_dict.get(cell, "?"), end=" ")
        print()
    print()


@jit(**JIT_ARGS)
def copy_state(source, target):
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
    return target


INFO_REWARD_IDX = 0
INFO_ALIVE_IDX = 1
INFO_LENGTH_IDX = 2
INFO_ROW_IDX = 3
INFO_COL_IDX = 4


@jit(**JIT_ARGS)
def get_info_from_state(state):
    """
    For each snake:
        0: reward
        1: alive
        2: length (final length if dead)
        3: head row
        4: head col
    """
    x = np.zeros(INFO_SHAPE, dtype="float32")
    x[:, INFO_REWARD_IDX] = state.rewards
    x[:, INFO_ALIVE_IDX] = state.alive
    x[:, INFO_LENGTH_IDX] = state.length
    for s in range(N_PLAYERS):
        r, c = state.head(s)
        x[s, INFO_ROW_IDX] = r
        x[s, INFO_COL_IDX] = c
    return x


@jit(**JIT_ARGS)
def get_nn_obs_from_state(state):
    """
    2 layers for each snake:
        - 1 if head, 0 otherwise
        - normalised body, increasing from tail
    -3: 1 if food, 0 otherwise
    -2: steps since hunger (normalised to 0-1)
    -1: current step (normalised to 0-1)
    """
    x = np.zeros(NN_OBS_SHAPE, dtype="float32")
    for s in range(N_PLAYERS):
        if state.alive[s]:
            idx_base = s * 2
            b = state.tail_ptr[s]
            val = 1.0 / GOOSE_MAX_LEN
            while True:
                r, c = state.snakes[s, b]
                x[idx_base + 1, r, c] = val
                val += 1.0 / GOOSE_MAX_LEN
                if b == state.head_ptr[s]:
                    x[idx_base, r, c] = 1.0
                    break
                b = (b + 1) % GOOSE_MAX_LEN
    x[-3, :, :] = state.snack_array
    x[-2, :, :] = (state.step % HUNGER_RATE) / HUNGER_RATE
    x[-1, :, :] = state.step / MAX_NUM_STEPS
    return x


@jit(**JIT_ARGS)
def moves_to_key(moves):
    key = 0
    for s in range(N_PLAYERS):
        key += MOVES ** s * moves[s]
    return key


key_to_move_array = np.zeros((N_PLAYERS ** MOVES, MOVES), dtype="int32")
for _moves in itertools.product(range(MOVES), repeat=N_PLAYERS):
    key = moves_to_key(np.array(_moves, dtype="int32"))
    key_to_move_array[key, :] = _moves


tree_spec = [
    ("parent", int32[:]),
    ("children", int32[:, :]),
    ("depth", int32[:]),
    ("moves", int32[:, :]),
    ("full_visits", float32[:, :]),
    ("forbidden", int32[:, :]),
    ("visits", float32[:, :, :]),
    ("values", float32[:, :, :]),
    ("probs", float32[:, :, :]),
    ("ptr", int32),
    ("size", int32),
    ("max_size", int32),
]


@jitclass(tree_spec)
class Tree:
    def __init__(self, max_size):
        self.parent = -np.empty(max_size, dtype="int32")
        self.children = np.empty((max_size, MOVES ** N_PLAYERS), dtype="int32")
        self.depth = np.empty(max_size, dtype="int32")
        self.moves = -np.empty((max_size, N_PLAYERS), dtype="int32")
        self.full_visits = np.empty((max_size, MOVES ** N_PLAYERS), dtype="float32")
        self.forbidden = np.empty((max_size, MOVES ** N_PLAYERS), dtype="int32")
        self.visits = np.empty((max_size, N_PLAYERS, MOVES), dtype="float32")
        self.values = np.empty((max_size, N_PLAYERS, MOVES), dtype="float32")
        self.probs = np.empty((max_size, N_PLAYERS, MOVES), dtype="float32")
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.parent[:] = -1
        self.children[:] = 0
        self.depth[:] = 0
        self.moves[:] = -1
        self.full_visits[:] = 0.0
        self.forbidden[:] = 0
        self.visits[:] = 0.0
        self.values[:] = 0.0
        self.probs[:] = 0.0
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


@jit(**JIT_ARGS)
def get_priors(state):
    """Mask illegal moves and almost mask collisions"""
    priors = np.ones((N_PLAYERS, MOVES), dtype="float32")
    for s in range(N_PLAYERS):
        if state.alive[s]:
            head_r, head_c = state.head(s)
            for m, (diff_r, diff_c) in enumerate(OFFSETS):
                if state.illegal_moves[s, m]:
                    priors[s, m] = 0.0
                    continue
                new_r, new_c = (head_r + diff_r) % N_ROWS, (head_c + diff_c) % N_COLS
                if state.snake_array[new_r, new_c]:
                    priors[s, m] = 0.00001
                mult = 1.0
                for s2 in range(N_PLAYERS):
                    if state.alive[s2]:
                        head2_r, head2_c = state.head(s2)
                        tail2_r, tail2_c = state.tail(s2)
                        if new_r == tail2_r and new_c == tail2_c:
                            mult *= 100_000.0  # cancel out the 0.00001
                priors[s, m] *= mult
        else:
            priors[s, 1:] = 0.0
    return priors


@jit(**JIT_ARGS)
def get_basic_scores(info):
    """Final scores for dead snakes, alive snakes are tied"""
    scores = np.zeros(N_PLAYERS, dtype="float32")
    ranked = np.argsort(info[:, 0])
    RANK_SCORES = np.linspace(-1, 1, N_PLAYERS).astype(np.float32)
    for rank in range(N_PLAYERS):
        s = ranked[rank]
        scores[s] = RANK_SCORES[rank]
    for rank in range(N_PLAYERS):
        s = ranked[rank]
        rew = info[s, 0]
        tied_idxs = np.where(info[:, 0] == rew)[0]
        n_tied = len(tied_idxs)
        if n_tied > 1:
            scores[tied_idxs] = scores[tied_idxs].mean()
    return scores


@jit(**JIT_ARGS)
def get_min_and_excess_scores(info):
    """Final scores for dead snakes, alive snakes get minimum, 'excess'
    score is what will eventually be allocated among living snakes
    """
    scores = np.zeros(N_PLAYERS, dtype="float32")
    ranked = np.argsort(info[:, 0])
    n_alive = int32(info[:, 1].sum())
    if n_alive > 0:
        alive_min_score = RANK_SCORES[N_PLAYERS - n_alive]
    for rank in range(N_PLAYERS):
        s = ranked[rank]
        if info[s, 1]:
            scores[s] = alive_min_score
        else:
            scores[s] = RANK_SCORES[rank]
    if n_alive <= N_PLAYERS - 2:  # at least two are dead
        for rank in range(N_PLAYERS):
            s = ranked[rank]
            if not info[s, 1]:
                rew = info[s, 0]
                tied_idxs = np.where(info[:, 0] == rew)[0]
                n_tied = len(tied_idxs)
                if n_tied > 1:
                    scores[tied_idxs] = scores[tied_idxs].mean()
    excess_score = -scores.sum()
    return scores, excess_score


@jit(**JIT_ARGS)
def get_prop_scores(info):
    """Split excess score according to snake length"""
    scores, excess_score = get_min_and_excess_scores(info)
    if excess_score > 0:
        lengths = info[:, 2] * info[:, 1]
        if lengths.sum() > 0:
            props = lengths / lengths.sum()
            scores += props * excess_score
    return scores


@jit(**JIT_ARGS)
def explore_moves(state, tree):
    """UCT-style MCTS inspired by AlphaZero"""
    Q = tree.values[tree.ptr] / (tree.visits[tree.ptr] + 1.0)
    P = tree.probs[tree.ptr] * get_priors(state)
    for s in range(N_PLAYERS):
        P[s] /= P[s].sum()  # renormalise P because we probably threw it off with priors
    U = (
        (tree.full_visits[tree.ptr].sum() + 1.0) ** 0.5
        / (tree.full_visits[tree.ptr] + 1.0)
        * EXPLORATION
    )
    for _key, _moves in enumerate(key_to_move_array):
        for s in range(N_PLAYERS):
            U[_key] *= P[s, _moves[s]]
        for s in range(N_PLAYERS):
            U[_key] += Q[s, _moves[s]]
    max_key = -1
    max_u = -999_999.0
    for _key, _u in enumerate(U):
        if tree.forbidden[tree.ptr, _key]:
            continue
        if _u > max_u:
            max_key = _key
            max_u = _u
    moves = -np.ones(MOVES, dtype="int32")
    if max_key >= 0:
        moves[:] = key_to_move_array[max_key]
    return moves


@jit(**JIT_ARGS)
def backup(tree, node_ptr, p, v):
    tree.ptr = node_ptr
    tree.probs[node_ptr] = p
    if node_ptr == 0:
        return
    key = moves_to_key(tree.moves[tree.ptr])
    tree.forbidden[tree.parent[node_ptr], key] = 0
    while True:
        parent_ptr = tree.parent[tree.ptr]
        if parent_ptr < 0:
            break
        key = moves_to_key(tree.moves[tree.ptr])
        tree.full_visits[parent_ptr, key] += 1.0
        for s in range(N_PLAYERS):
            m = tree.moves[tree.ptr][s]
            tree.visits[parent_ptr, s, m] += 1.0
            tree.values[parent_ptr, s, m] += v[s]
        tree.get_parent()


@jit(**JIT_ARGS)
def simulate(state, tree, signal_sp, I_sp, X_sp, sp_ptr):
    """Explores new nodes and stores for evaluation. Returns True if done."""
    if tree.probs[0].sum() <= 0.01:  # i.e. if tree is empty, eval first node
        signal_sp[sp_ptr] = tree.ptr
        I_sp[sp_ptr] = get_info_from_state(state)
        X_sp[sp_ptr] = get_nn_obs_from_state(state)
        return True
    state = copy_state(state, State())
    while True:
        moves = explore_moves(state, tree)
        if moves[0] == -1:  # i.e. no nodes available
            tree.ptr = 0
            return True
        has_child = tree.get_child(moves)
        if not has_child:
            success = tree.make_child(moves)
            if not success:  #  i.e. tree is full
                tree.ptr = 0
                return True
        state.transition(moves)
        if state.done or not has_child:
            signal_sp[sp_ptr] = tree.ptr
            key = moves_to_key(moves)
            tree.forbidden[tree.parent[tree.ptr], key] = 1
            I_sp[sp_ptr] = get_info_from_state(state)
            X_sp[sp_ptr] = get_nn_obs_from_state(state)
            break
    tree.ptr = 0
    return False


env_done = np.zeros(ENVS, dtype="bool")

# self-play data (for evaluating moves)
signal_sp = -np.ones(ENVS * NODES_PER_SEARCH_STEP, dtype="int32")  # stores node indexes
I_sp = np.zeros((ENVS * NODES_PER_SEARCH_STEP, *INFO_SHAPE), dtype="int32")
X_sp = np.zeros((ENVS * NODES_PER_SEARCH_STEP, *NN_OBS_SHAPE), dtype="float32")
P_sp = np.zeros((ENVS * NODES_PER_SEARCH_STEP, N_PLAYERS, MOVES), dtype="float32")
V_sp = np.zeros((ENVS * NODES_PER_SEARCH_STEP, N_PLAYERS), dtype="float32")

# training data ('real' moves that will go into the buffer)
signal_tr = np.zeros(ENVS * MAX_NUM_STEPS, dtype="int32")  # obs to be added to buff
I_tr = np.zeros((ENVS * MAX_NUM_STEPS, *INFO_SHAPE), dtype="int32")
X_tr = np.zeros((ENVS * MAX_NUM_STEPS, *NN_OBS_SHAPE), dtype="float32")
P_tr = np.zeros((ENVS * MAX_NUM_STEPS, N_PLAYERS, MOVES), dtype="float32")
V_tr = np.zeros((ENVS * MAX_NUM_STEPS, N_PLAYERS), dtype="float32")

# buffer data (contains multiple games' worth of data)
buffer_ptr = 0  # marks insertion point for buffer
I_bf = np.zeros((BUFFER_SIZE, *INFO_SHAPE), dtype="int32")
X_bf = np.zeros((BUFFER_SIZE, *NN_OBS_SHAPE), dtype="float32")
P_bf = np.zeros((BUFFER_SIZE, N_PLAYERS, MOVES), dtype="float32")
V_bf = np.zeros((BUFFER_SIZE, N_PLAYERS), dtype="float32")


TREES = typed.List()
for _ in range(ENVS):
    tree_size = NODES_PER_SEARCH_STEP * SELFPLAY_SEARCH_ROUNDS
    TREES.append(Tree(tree_size))


STATES = typed.List()
for _ in range(ENVS):
    STATES.append(State())


@jit(**JIT_ARGS)
def initialise_envs(STATES, TREES, env_done):
    for tree in TREES:
        tree.reset()
    for state in STATES:
        state.reset()
    env_done[:] = False


@jit(**JIT_ARGS, parallel=True)
def collect_search_data(states, trees, signal_sp, I_sp, X_sp, env_done):
    for e in prange(ENVS):
        if env_done[e]:
            continue
        state = states[e]
        tree = trees[e]
        for n in range(NODES_PER_SEARCH_STEP):
            sp_ptr = e * NODES_PER_SEARCH_STEP + n
            is_done = simulate(state, tree, signal_sp, I_sp, X_sp, sp_ptr)
            if is_done:
                break


# @jit(**JIT_ARGS)
def evaluate_search_data(I_sp, X_sp, P_sp, V_sp):
    # Replaced with P_sp, V_sp = net(X_sp), below is just a placeholder
    """
    for sp_ptr in range(len(I_sp)):
        if I_sp[sp_ptr].sum() > 0.0:
            V_sp[sp_ptr] = get_prop_scores(I_sp[sp_ptr])
    P_sp[:] = 0.25
    """
    states = torch.from_numpy(X_sp).to(device=DEVICE, dtype=torch.float32)
    head_locs = torch.from_numpy(I_sp[..., INFO_ROW_IDX] * N_COLS + I_sp[..., INFO_COL_IDX]).to(device=DEVICE,
                                                                                                dtype=torch.int64)
    still_alive = torch.from_numpy(I_sp[..., INFO_ALIVE_IDX]).to(device=DEVICE, dtype=torch.bool)
    with torch.no_grad():
        logits, values = MODEL(states, head_locs, still_alive)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        values = values.cpu().numpy()

    # Score the dead geese
    agent_rankings = stats.rankdata(I_sp[..., INFO_REWARD_IDX], method='average', axis=-1) - 1.
    agent_rankings_rescaled = 2. * agent_rankings / (N_PLAYERS - 1.) - 1.
    final_values = np.where(
        I_sp[..., INFO_ALIVE_IDX],
        values,
        agent_rankings_rescaled
    )

    P_sp[:] = probs
    V_sp[:] = final_values


@jit(**JIT_ARGS, parallel=True)
def backup_search_data(trees, signal_sp, P_sp, V_sp, env_done):
    for e in prange(ENVS):
        if env_done[e]:
            continue
        tree = trees[e]
        for n in range(NODES_PER_SEARCH_STEP):
            sp_ptr = e * NODES_PER_SEARCH_STEP + n
            node_ptr = signal_sp[sp_ptr]
            if node_ptr > -1:
                p = P_sp[sp_ptr]
                v = V_sp[sp_ptr]
                backup(tree, node_ptr, p, v)
                signal_sp[sp_ptr] = -1


@jit(**JIT_ARGS, parallel=True)
def save_step_and_advance(states, trees, signal_tr, I_tr, X_tr, P_tr, env_done):
    for e in prange(ENVS):
        if env_done[e]:
            continue
        state = states[e]
        tree = trees[e]
        tr_ptr = e * MAX_NUM_STEPS + state.step
        signal_tr[tr_ptr] = 1
        I_tr[tr_ptr] = get_info_from_state(state)
        X_tr[tr_ptr] = get_nn_obs_from_state(state)
        p = tree.visits[0].copy()
        for s in range(N_PLAYERS):
            p[s] /= p[s].sum()
        P_tr[tr_ptr] = p
        moves = np.zeros(MOVES, dtype="int32")
        for s in range(N_PLAYERS):
            moves[s] = np.argmax(p[s])
        state.transition(moves)
        env_done[e] = state.done
        tree.reset()


def clear_selfplay_data(signal_sp, I_sp, X_sp, P_sp, V_sp):
    signal_sp[:] = -1
    I_sp[:] = 0.0
    X_sp[:] = 0.0
    P_sp[:] = 0.0
    V_sp[:] = 0.0


@jit(**JIT_ARGS, parallel=True)
def finalise_values_for_training_data(states, V_tr):
    for e in prange(ENVS):
        state = states[e]
        final_values = get_basic_scores(get_info_from_state(state))
        tr_slice_start = e * MAX_NUM_STEPS
        tr_slice_end = (e + 1) * MAX_NUM_STEPS
        V_tr[tr_slice_start:tr_slice_end] = final_values


@jit(**JIT_ARGS)
def insert_training_data_into_buffer(
    buffer_ptr, I_bf, X_bf, P_bf, V_bf, signal_tr, I_tr, X_tr, P_tr, V_tr
):
    for tr_ptr in range(len(signal_tr)):
        if not signal_tr[tr_ptr]:
            continue
        I_bf[buffer_ptr] = I_tr[tr_ptr]
        X_bf[buffer_ptr] = X_tr[tr_ptr]
        P_bf[buffer_ptr] = P_tr[tr_ptr]
        V_bf[buffer_ptr] = V_tr[tr_ptr]
        buffer_ptr = (buffer_ptr + 1) % BUFFER_SIZE
    signal_tr[:] = 0
    I_tr[:] = 0.0
    X_tr[:] = 0.0
    P_tr[:] = 0.0
    V_tr[:] = 0.0
    return buffer_ptr


def get_state_dict_from_I_X_P_V(I, X, P, V):
    state_dict_list = []
    if OBS_TYPE == ge.ObsType.COMBINED_GRADIENT_OBS:
        """
        Expects a tensor of shape (3 + 2*n_players, 7, 11)
        The channels contain the following information about each cell of the board:
        for i in range (n_players):
            * contains_head[i]
            * contains_body[i], where the value of this cell represents how close to the tail this cell is
                Values of 1 / GOOSE_MAX_LEN represent the tail, and n / GOOSE_MAX_LEN represents the nth element
                counting from the tail, and INCLUDING the head
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        geese = []
        idx_array = np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
        for i in range(N_PLAYERS):
            geese.append([])
            if X[2 * i].sum() == 0:
                continue
            for j in range(1, GOOSE_MAX_LEN):
                mask = X[2 * i + 1] == float(j) / GOOSE_MAX_LEN
                idx = idx_array[mask]
                geese[i].append(idx.item())
                if X[2 * i][mask] == 1:
                    break
            geese[i].reverse()
            if j >= GOOSE_MAX_LEN - 1:
                print(f'ERROR: Invalid goose length {j}, {geese[i]}')
                return None
        food = [f.item() for f in idx_array[X[-3] == 1]]
        if len(food) != 2:
            print(f'ERROR: Invalid food locations {food}, {X[-3]}')
            return None
        agent_rankings = stats.rankdata(V, method='average') - 1.
        for i in range(N_PLAYERS):
            dict_i = {
                # 'action': 'NA',
                'reward': I[i, INFO_REWARD_IDX].item(),
                # 'info': {},
                'observation': {
                    # 'remainingOverageTime' is not computed and is included only for compatibility
                    'index': i
                },
                'status': 'ACTIVE' if I[i, INFO_ALIVE_IDX] == 1. else 'DONE',
                'final_rank': agent_rankings[i].item(),
            }
            if dict_i['status'] == 'ACTIVE':
                dict_i.update({
                    'available_actions_mask': [1] * 4,
                    'policy': [p for p in P[i].astype(float)]
                })
            if i == 0:
                dict_i['observation'].update({
                    'step': np.round(X[-1, 0, 0] * MAX_NUM_STEPS).astype(int).item(),
                    'geese': geese,
                    'food': food
                })
            state_dict_list.append(dict_i)
    else:
        raise ValueError(f'Unsupported obs_type: {OBS_TYPE}')
    return state_dict_list


def save_buffer_to_disk(I_bf, X_bf, P_bf, V_bf, save_steps_batch_queue):
    global buffer_ptr
    all_states = []
    for i in range(buffer_ptr):
        if np.isnan(P_bf).any():
            print('ERROR: NaN policy')
            continue
        state_dict = get_state_dict_from_I_X_P_V(
            I_bf[i],
            X_bf[i],
            P_bf[i],
            V_bf[i]
        )
        if state_dict is not None:
            all_states.append(state_dict)

    for i in range(0, len(all_states), 200):
        steps_batch = all_states[i:i+200]
        save_steps_batch_queue.put_nowait(steps_batch)
    buffer_ptr = 0


class Stopwatch:
    def __init__(self, use_stopwatch=True):
        """Used to time function calls (can't profile numba)"""
        self.use_stopwatch = use_stopwatch
        self.durations = dict()
        self.active_key = None
        self._start_time = None

    def __repr__(self):
        s = "Stopwatch:\n"
        s += "\n".join([f"  {k}: {v:.3f}" for k, v in self.durations.items()])
        return s

    def start(self, key):
        if not self.use_stopwatch:
            return

        if self.active_key is not None:
            self.stop()
        self._start_time = time()
        self.active_key = key

    def stop(self):
        if not self.use_stopwatch:
            return

        old_time = self.durations.get(self.active_key, 0)
        diff = time() - self._start_time
        new_time = old_time + diff
        self.durations[self.active_key] = new_time
        self.active_key = None


def selfplay_loop(save_steps_batch_queue, stopwatch):
    global buffer_ptr

    stopwatch.start("initialise_envs")
    initialise_envs(STATES, TREES, env_done)

    pbar = tqdm(total=MAX_NUM_STEPS * ENVS, desc=f'Steps in {ENVS} environments')
    while not env_done.all():
        for _ in range(SELFPLAY_SEARCH_ROUNDS):

            stopwatch.start("collect_search_data")
            collect_search_data(STATES, TREES, signal_sp, I_sp, X_sp, env_done)

            stopwatch.start("evaluate_search_data")
            evaluate_search_data(I_sp, X_sp, P_sp, V_sp)

            stopwatch.start("backup_search_data")
            backup_search_data(TREES, signal_sp, P_sp, V_sp, env_done)
            # TODO: Evaluated probabilities will just get attached to a single
            # node and be static after that, which will hardcode values from a
            # particular random food position. Unsure if there is a way around it

        stopwatch.start("save_step_and_advance")
        save_step_and_advance(STATES, TREES, signal_tr, I_tr, X_tr, P_tr, env_done)

        # comment/uncomment to see the playout for one of the envs
        # display(STATES[0])

        stopwatch.start("clear_selfplay_data")
        clear_selfplay_data(signal_sp, I_sp, X_sp, P_sp, V_sp)
        pbar.update(ENVS)
    pbar.close()

    stopwatch.start("finalise_values_for_training_data")
    finalise_values_for_training_data(STATES, V_tr)

    stopwatch.start("insert_training_data_into_buffer")
    buffer_ptr = insert_training_data_into_buffer(
        buffer_ptr, I_bf, X_bf, P_bf, V_bf, signal_tr, I_tr, X_tr, P_tr, V_tr
    )

    stopwatch.start("save_buffer_to_disk")
    save_buffer_to_disk(I_bf, X_bf, P_bf, V_bf, save_steps_batch_queue)

    stopwatch.stop()


def start_selfplay_loop(
        model: nn.Module,
        device: torch.device,
        dataset_dir: Path,
        weights_dir: Path,
        max_saved_batches: int,
        obs_type: ge.ObsType = ge.ObsType.COMBINED_GRADIENT_OBS,
        allow_resume: bool = False
):
    # No-op stopwatch
    stopwatch = Stopwatch(use_stopwatch=False)

    # NN global vars
    global DEVICE, OBS_TYPE, MODEL
    DEVICE = device
    OBS_TYPE = obs_type
    MODEL = model

    MODEL.to(device=DEVICE)
    MODEL.load_state_dict(torch.load('/home/isaiah/GitHub/Kaggle/Hungry_Geese/cp.pt', map_location=DEVICE))
    MODEL.eval()

    # Check that dataset_dir exists and is empty
    if dataset_dir.exists() and any(list(dataset_dir.iterdir())):
        if not allow_resume:
            raise RuntimeError(f'dataset_dir already exists and is not empty: {dataset_dir}')
        # Check that the directory only contains replay data files
        all_files = [f for f in dataset_dir.iterdir() if not f.stem.startswith('.')]
        if not all([f.suffix == '.ljson' for f in all_files]):
            raise RuntimeError(f'dataset_dir already exists and contains non-".ljson" files: {dataset_dir}')
        # Check that the directory does not contain more replay data files than allowed by max_saved_batches
        all_files.sort(key=lambda f: int(f.stem))
        if int(all_files[-1].stem) >= max_saved_batches:
            raise RuntimeError(f'dataset_dir already exists and contains files with id > max_saved_batches: '
                               f'{dataset_dir} - {all_files[-1].stem} > {max_saved_batches}')
        # Find the index to restart at
        all_files.sort(key=lambda f: f.stat().st_mtime)
        start_idx = (int(all_files[-1].stem) + 1) % max_saved_batches
        print(f'Resuming data generation. Latest replay file: {all_files[-1].name}')
    else:
        start_idx = 0
    dataset_dir.mkdir(exist_ok=True)

    # Start the worker who will save batches of experience to disk
    save_steps_batch_queue = mp.Queue()
    save_steps_to_disk_worker = mp.Process(
        target=save_episodes_worker,
        args=(dataset_dir, save_steps_batch_queue, max_saved_batches, start_idx),
    )
    save_steps_to_disk_worker.daemon = True
    save_steps_to_disk_worker.start()

    # Logging
    print(f'Running {ENVS} environments across {NUMBA_THREADS} threads.\n'
          f'For each step, exploring {NODES_PER_SEARCH_STEP} nodes {SELFPLAY_SEARCH_ROUNDS} times, '
          f'for a total of {NODES_PER_SEARCH_STEP * SELFPLAY_SEARCH_ROUNDS} rollouts.')

    current_weights_path = None
    while True:
        selfplay_loop(save_steps_batch_queue, stopwatch)
        current_weights_path = reload_model_weights(model, weights_dir, current_weights_path, DEVICE)


if __name__ == "__main__":
    DEVICE = torch.device('cuda')
    OBS_TYPE = ge.ObsType.COMBINED_GRADIENT_OBS

    n_channels = 128
    activation = nn.ReLU
    model_kwargs = dict(
        block_class=models.BasicConvolutionalBlock,
        conv_block_kwargs=[
            dict(
                in_channels=OBS_TYPE.get_obs_spec()[-3],
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
        # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
    )
    MODEL = models.FullConvActorCriticNetwork(**model_kwargs)
    MODEL.to(device=DEVICE)
    reload_model_weights(MODEL, Path('/home/isaiah/GitHub/Kaggle/Hungry_Geese/'), None, DEVICE)
    MODEL.eval()

    _save_steps_batch_queue = mp.Queue()

    _stopwatch = Stopwatch()
    selfplay_loop(_save_steps_batch_queue, _stopwatch)  # first run will be slower because JIT

    for _ in tqdm(range(2)):
        selfplay_loop(_save_steps_batch_queue, _stopwatch)

    print(_stopwatch)
    print(f"threading layer chosen: {threading_layer()}")
