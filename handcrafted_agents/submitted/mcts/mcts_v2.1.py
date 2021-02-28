import copy
from enum import *
from kaggle_environments import make as kaggle_make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, histogram, translate
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col as _row_col
from numbers import Number
import numpy as np
from random import sample
from scipy import stats
import time
from typing import *

HUNGER_RATE = 40.
MAX_NUM_STEPS = 200.
GOOSE_MAX_LEN = 99.
N_ROWS = 7
N_COLS = 11


class ActionMasking(Enum):
    """
    Available action masking settings returned by info_dict['available_actions_mask'].
    NONE: No action masking
    OPPOSITE: Only mask actions that are the opposite of the last action taken
    LETHAL: Mask opposite actions and any action that would result in moving into a square with a goose body
        (still allows movement into squares with goose heads for geese of length 1 or tails for any goose)
        Warning: LETHAL action masking will sometimes result in no available actions - take this into account
    """
    NONE = auto()
    OPPOSITE = auto()
    LETHAL = auto()

    def get_action_mask(self, state: List[Dict]) -> np.ndarray:
        available_actions_mask = np.ones((len(state), 4), dtype=np.bool)
        if self == ActionMasking.NONE:
            pass
        elif self == ActionMasking.OPPOSITE:
            if state[0]['observation']['step'] > 0:
                for agent_idx, agent in enumerate(state):
                    last_action = Action[agent['action']]
                    banned_action_idx = tuple(Action).index(last_action.opposite())
                    available_actions_mask[agent_idx, banned_action_idx] = False
        elif self == ActionMasking.LETHAL:
            if state[0]['observation']['step'] > 0:
                all_goose_locs = []
                for goose_loc_list in state[0]['observation']['geese']:
                    # Ignore geese that are only a head
                    if len(goose_loc_list) > 1:
                        # Don't mask the tail position
                        all_goose_locs += [row_col(n) for n in goose_loc_list[:-1]]
                for agent_idx, agent in enumerate(state):
                    goose_loc_list = state[0]['observation']['geese'][agent_idx]
                    if len(goose_loc_list) > 0:
                        last_action = Action[agent['action']]
                        banned_action_idx = tuple(Action).index(last_action.opposite())
                        available_actions_mask[agent_idx, banned_action_idx] = False
                        head_loc = np.array(row_col(goose_loc_list[0]))
                        for act in tuple(Action):
                            destination = head_loc + np.array(act.to_row_col())
                            destination[0] = (N_ROWS + destination[0]) % N_ROWS
                            destination[1] = (N_COLS + destination[1]) % N_COLS
                            if tuple(destination) in all_goose_locs:
                                banned_action_idx = tuple(Action).index(act)
                                available_actions_mask[agent_idx, banned_action_idx] = False
        else:
            raise ValueError(f'ActionMasking not yet implemented: {self.name}')

        return available_actions_mask


def row_col(position: int) -> Tuple[int, int]:
    return _row_col(position, N_COLS)


def rowwise_random_choice(p: np.ndarray) -> np.ndarray:
    """
    Numpy has no built-in way to take a random choice over a 2D array
    This implementation is based on the idea of inverse transform sampling

    Returns:
        a: The index of the sampled element from each row, with shape (p.shape[0],)
    """
    assert p.ndim == 2 and np.all(p >= 0.)
    p_norm = p / p.sum(axis=1, keepdims=True)
    return np.argmax(
        np.cumsum(p_norm, axis=1) > np.random.random_sample((p_norm.shape[0], 1)),
        axis=1
    )


def print_array_one_line(arr: Union[np.ndarray, Number]) -> str:
    if type(arr) == np.ndarray:
        return '[' + ', '.join([print_array_one_line(a) for a in arr]) + ']'
    else:
        return f'{arr:.2f}'


ACTIONS_TUPLE = tuple(Action)


class LightweightEnv:
    """
    Mostly the same as Kaggle hungry_geese env, but faster to work with and copy
    """
    def __init__(self, configuration: Configuration, debug: bool = False):
        self.configuration = configuration
        self.n_rows = configuration.rows
        self.n_cols = configuration.columns
        self.debug = debug

        self.agent_count = None
        self.geese = None
        self.food = None
        self.last_actions = None
        self.step_counter = None
        self.rewards = None
        self.steps = None

        self.reset()

    def reset(self, num_agents: int = 4) -> List[Dict]:
        self.agent_count = num_agents
        heads = sample(range(self.n_cols * self.n_rows), self.agent_count)
        self.geese = [[head] for head in heads]
        food_candidates = set(range(self.n_cols * self.n_rows)).difference(heads)
        # Ensure we only place as many food as there are open squares
        min_food = min(self.configuration.min_food, len(food_candidates))
        self.food = sample(food_candidates, min_food)
        self.last_actions = [Action.NORTH for _ in range(self.agent_count)]
        self.step_counter = 0
        self.rewards = [0 for _ in range(self.agent_count)]
        self.steps = []
        self.generate_and_append_next_state()

        return self.steps[-1]

    def step(self, actions: Union[List[str], Sequence[int], np.ndarray]):
        assert not self.done

        if type(actions) == np.ndarray:
            actions = actions.ravel()
        assert len(actions) == self.agent_count, f'Got {len(actions)} actions for {self.agent_count} agents'
        if type(actions[0]) != str:
            actions = [ACTIONS_TUPLE[i].name for i in actions]

        for index, goose in enumerate(self.geese):
            if len(goose) == 0:
                continue
            action = Action[actions[index]]

            # Check action direction on any step after the first
            if self.last_actions[index] == action.opposite() and self.step_counter > 0:
                self.debug_print(f'Opposite action: {index, action, self.last_actions[index]}')
                self.geese[index] = []
                continue
            self.last_actions[index] = action

            head = translate(goose[0], action, self.n_cols, self.n_rows)

            # Consume food or drop a tail piece
            if head in self.food:
                self.food.remove(head)
            else:
                goose.pop()

            # Self collision
            if head in goose:
                self.debug_print(f'Body Hit: {index, action, head, goose}')
                self.geese[index] = []
                continue

            while len(goose) >= self.configuration.max_length:
                # Free a spot for the new head if needed
                goose.pop()
            # Add New Head to the Goose
            goose.insert(0, head)

            # If hunger strikes, remove from the tail
            if self.step_counter % self.configuration.hunger_rate == 0 and self.step_counter > 0:
                if len(goose) > 0:
                    goose.pop()
                if len(goose) == 0:
                    self.debug_print(f'Goose Starved: {index}')
                    continue

        goose_positions = histogram(
            position
            for goose in self.geese
            for position in goose
        )

        # Check for collisions
        for index, goose in enumerate(self.geese):
            if len(goose) > 0:
                head = self.geese[index][0]
                if goose_positions[head] > 1:
                    self.debug_print(f'Goose Collision: {index, self.last_actions[index]}')
                    self.geese[index] = []

        # Add food if min_food threshold reached
        needed_food = self.configuration.min_food - len(self.food)
        if needed_food > 0:
            collisions = {
                position
                for goose in self.geese
                for position in goose
            }
            available_positions = set(range(self.n_rows * self.n_cols)).difference(collisions).difference(self.food)
            # Ensure we don't sample more food than available positions
            needed_food = min(needed_food, len(available_positions))
            self.food.extend(sample(available_positions, needed_food))

        self.step_counter += 1
        # Set rewards after deleting all geese to ensure that geese don't receive a reward on the turn they perish
        for index, goose in enumerate(self.geese):
            if len(goose) > 0:
                # Adding 1 to len(env.steps) ensures that if an agent gets reward 4507, it died on turn 45 with length 7
                self.rewards[index] = (self.step_counter + 1) * (self.configuration.max_length + 1) + len(goose)

        self.generate_and_append_next_state()
        return self.steps[-1]

    def generate_and_append_next_state(self) -> NoReturn:
        state_dict_list = []
        statuses = self.get_statuses()
        for i in range(self.agent_count):
            dict_i = {
                'action': self.last_actions[i].name,
                'reward': self.rewards[i],
                'info': {},
                'observation': {
                    # 'remainingOverageTime' is not computed and is included only for compatibility
                    'index': i
                },
                'status': statuses[i]
            }
            if i == 0:
                dict_i['observation'].update({
                    'step': self.step_counter,
                    'geese': self.geese,
                    'food': self.food
                })
            state_dict_list.append(dict_i)

        self.steps.append(state_dict_list)

    @property
    def state(self) -> List[Dict]:
        return self.steps[-1]

    @property
    def done(self) -> bool:
        n_geese_alive = len([True for goose in self.geese if len(goose) > 0])
        return n_geese_alive <= 1

    def get_statuses(self) -> List[str]:
        if self.done:
            return ['DONE' for _ in range(self.agent_count)]
        else:
            return ['ACTIVE' if len(goose) > 0 else 'DONE' for goose in self.geese]

    def clone(self):
        cloned_env = LightweightEnv(self.configuration)

        cloned_env.agent_count = self.agent_count
        cloned_env.geese = copy.deepcopy(self.geese)
        cloned_env.food = copy.copy(self.food)
        cloned_env.last_actions = copy.copy(self.last_actions)
        cloned_env.step_counter = self.step_counter
        cloned_env.rewards = copy.copy(self.rewards)
        cloned_env.steps = copy.deepcopy(self.steps)

        return cloned_env

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        return self.clone()

    def canonical_string_repr(self, include_food=True) -> str:
        if self.done:
            raise RuntimeError('Environment has finished')
        else:
            canonical_string = ''
            canonical_string += f'S: {self.step_counter} '
            if include_food:
                canonical_string += 'F: ' + '_'.join([str(f) for f in self.food]) + ' '
            canonical_string += 'G: '
            for index, goose, status in zip(range(self.agent_count), self.geese, self.get_statuses()):
                canonical_string += f'{index}_'
                if status == 'DONE':
                    canonical_string += f'D '
                else:
                    canonical_string += '_'.join([str(g) for g in goose]) + f'_{self.last_actions[index].value} '

            return canonical_string

    def debug_print(self, out: str):
        if self.debug:
            print(out)

    def render_ansi(self) -> str:
        food_symbol = "F"
        column_divider = "|"
        row_divider = "+" + "+".join(["---"] * self.n_cols) + "+\n"

        board = [" "] * (self.n_rows * self.n_cols)
        for pos in self.food:
            board[pos] = food_symbol

        for index, goose in enumerate(self.geese):
            for position in goose:
                board[position] = str(index)

        out = row_divider
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                out += column_divider + f" {board[(row * self.n_cols) + col]} "
            out += column_divider + "\n" + row_divider

        return out


def make(environment: str, debug: bool = False, **kwargs) -> LightweightEnv:
    assert environment == 'hungry_geese'
    config = kaggle_make(environment, debug=debug, **kwargs).configuration
    return LightweightEnv(config, debug=debug)


def make_from_state(state: Observation,
                    last_actions: List[Action],
                    configuration: Optional[Configuration] = None,
                    *args, **kwargs) -> LightweightEnv:
    if configuration is None:
        configuration = kaggle_make('hungry_geese').configuration
    configuration = Configuration(configuration)
    env = LightweightEnv(configuration, *args, **kwargs)

    env.agent_count = len(state.geese)
    env.geese = copy.deepcopy(state.geese)
    env.food = copy.copy(state.food)
    env.last_actions = copy.copy(last_actions)
    env.step_counter = state.step

    rewards = [0.] * len(state.geese)
    for index, goose in enumerate(state.geese):
        if len(goose) > 0:
            # Adding 1 to len(env.steps) ensures that if an agent gets reward 4507, it died on turn 45 with length 7
            rewards[index] = (state.step + 1) * (configuration.max_length + 1) + len(goose)
    env.rewards = rewards

    env.steps = [None] * state.step
    env.generate_and_append_next_state()

    return env


class Node:
    def __init__(
            self,
            geese_still_playing_mask: Sequence[bool],
            available_actions_masks: Optional[np.ndarray],
            initial_policies: Optional[np.ndarray]
    ):
        self.geese_still_playing = np.array(geese_still_playing_mask)
        self.n_geese = len(self.geese_still_playing)
        assert self.geese_still_playing.any()
        if available_actions_masks is None:
            self.available_actions_masks = np.ones_like(initial_policies)
        else:
            self.available_actions_masks = available_actions_masks
        self.initial_policies = initial_policies
        assert self.initial_policies.shape == (self.n_geese, 4)
        assert np.all(0. <= self.initial_policies) and np.all(self.initial_policies <= 1.)
        assert np.allclose(self.initial_policies.sum(axis=-1), 1.)
        # Re-normalize policy distribution
        self.initial_policies = self.initial_policies * available_actions_masks
        if np.any(self.initial_policies.sum(axis=1) == 0.):
            if np.logical_and(self.initial_policies.sum(axis=1) == 0.,
                              self.available_actions_masks.any(axis=1)).any():
                print('WARNING: All available actions have 0% probability')
            self.initial_policies = np.where(
                self.initial_policies.sum(axis=1, keepdims=True) == 0.,
                0.25,
                self.initial_policies
            )
        self.initial_policies = self.initial_policies / self.initial_policies.sum(axis=1, keepdims=True)

        self.q_vals = np.zeros_like(self.initial_policies)
        self.n_visits = np.zeros_like(self.initial_policies)

    def update(self, actions: np.ndarray, values: np.ndarray) -> NoReturn:
        if actions.shape != (self.n_geese,):
            raise RuntimeError(f'Actions should be of shape {(self.n_geese,)}, got {actions.shape}')
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape != (self.n_geese, 1):
            raise RuntimeError(f'Values should be of shape {(self.n_geese, 1)}, got {values.shape}')
        if not np.isclose(values.sum(), 0.):
            raise RuntimeError(f'Values should sum to 0, got {values.ravel()} which sums to {values.sum()}')
        if (values.ravel()[self.geese_still_playing].min(initial=float('inf')) <=
                values.ravel()[~self.geese_still_playing].max(initial=float('-inf'))):
            raise RuntimeError(f'Values for dead geese should always be less than those for still living geese.\n'
                               f'Values:\n{values.ravel()}\n'
                               f'Geese still playing:\n{self.geese_still_playing}\n')

        selected_actions_mask = np.eye(4)[actions]
        self.q_vals = np.where(
            selected_actions_mask,
            (self.n_visits * self.q_vals + values) / (self.n_visits + 1.),
            self.q_vals
        )
        self.n_visits += selected_actions_mask

    def get_puct_actions(self, c_puct: float) -> np.ndarray:
        uct_vals = self.q_vals + (c_puct *
                                  self.initial_policies *
                                  np.sqrt(self.n_visits.sum(axis=1, keepdims=True)) / (1. + self.n_visits))
        uct_vals = np.where(
            self.available_actions_masks,
            uct_vals,
            -100.
        )
        """
        We could just go from here using a simple np.argmax(uct_vals, axis=1) to select actions,
        but this would always tiebreak towards actions with lower indices.
        Therefore, we sample from all actions whose uct value equals the max uct value for that agent.
        """
        uct_max = uct_vals.max(axis=1, keepdims=True)
        return np.where(
            self.geese_still_playing,
            rowwise_random_choice((uct_vals == uct_max).astype(np.float)),
            0
        )

    def get_improved_policies(self, temp: float = 1.) -> np.ndarray:
        assert temp >= 0.
        if temp == 0.:
            max_n_visits = self.n_visits.max(axis=1, keepdims=True)
            probs = np.where(
                self.n_visits == max_n_visits,
                1.,
                0.
            )
        else:
            probs = np.power(self.n_visits, 1. / temp)
        return probs / probs.sum(axis=1, keepdims=True)

    def get_improved_actions(self, temp: float = 1.) -> np.ndarray:
        probs = self.get_improved_policies(temp)
        return np.where(
            self.geese_still_playing,
            rowwise_random_choice(probs),
            0
        )


class BasicMCTS:
    def __init__(
            self,
            action_mask_func: Callable,
            actor_critic_func: Callable,
            terminal_value_func: Callable,
            c_puct: float = np.sqrt(2.),
            include_food: bool = False,
    ):
        self.action_mask_func = action_mask_func
        self.actor_critic_func = actor_critic_func
        self.terminal_value_func = terminal_value_func
        self.c_puct = c_puct
        self.include_food = include_food
        self.nodes = {}

    def _search(self, env: LightweightEnv) -> np.ndarray:
        """
        This function performs one iteration of MCTS. It is recursively called
        until a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the actor_critic function is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path.

        @return: the values of the current state
        """
        if env.done:
            return self.terminal_value_func(env.state)

        s = env.canonical_string_repr(include_food=self.include_food)
        node = self.nodes.get(s, None)
        if node is None:
            # Leaf node
            full_state = env.state
            policy_est, value_est = self.actor_critic_func(full_state)
            self.nodes[s] = Node(
                [status != 'DONE' for status in env.get_statuses()],
                self.action_mask_func(full_state),
                policy_est
            )
            return value_est

        a = node.get_puct_actions(self.c_puct).ravel()
        env.step(a)
        v = self._search(env)
        node.update(a, v)
        return v

    def run_mcts(
            self,
            env: LightweightEnv,
            n_iter: int,
            max_time: float = float('inf'),
    ) -> Node:
        start_time = time.time()
        for _ in range(n_iter):
            if time.time() - start_time >= max_time:
                break
            self._search(env.clone())

        return self.nodes[env.canonical_string_repr(include_food=self.include_food)]


"""
"Imported" the following packages:
config
utils
lightweight_env
basic_mcts
"""
from scipy import special

BOARD_DIMS = np.array([N_ROWS, N_COLS])


def wrap(position: np.ndarray) -> np.ndarray:
    assert position.shape == (2,), f'{position.shape}'
    return (position + BOARD_DIMS) % BOARD_DIMS


def torus_manhattan(a: np.ndarray, b: np.ndarray) -> int:
    dists = np.stack([
        np.abs(a - b),
        (a + BOARD_DIMS) - b,
        (b + BOARD_DIMS) - a
    ]).min(axis=0)
    return dists.sum()


# Precompute directions_dict for get_direction function
DIRECTIONS_DICT = {tuple(wrap(np.array(act.to_row_col()))): act for act in Action}

# Precompute distance matrix on startup
DISTANCE_MATRIX = {}
for x in range(N_ROWS):
    for y in range(N_COLS):
        key = (x, y)
        DISTANCE_MATRIX[key] = {}
        for x_prime in range(N_ROWS):
            for y_prime in range(N_COLS):
                key_prime = (x_prime, y_prime)
                DISTANCE_MATRIX[key][key_prime] = torus_manhattan(np.array(key), np.array(key_prime))

# Precompute action/delta_location_np pairs
MOVES = tuple([(a, np.array(a.to_row_col())) for a in Action])


def get_direction(from_loc: np.ndarray, to_loc: np.ndarray) -> Action:
    return DIRECTIONS_DICT[tuple(wrap(to_loc - from_loc))]


def action_mask_func(state):
    return ActionMasking.LETHAL.get_action_mask(state)


# No need to account for illegal actions here, as this is accounted for later using action masking + renormalization
def actor_func(state):
    food_locs = [row_col(f) for f in state[0]['observation']['food']]
    food_distances = np.zeros((len(state), 4))
    for goose_index, goose in enumerate(state[0]['observation']['geese']):
        if len(goose) > 0:
            for action_index, (_, m) in enumerate(MOVES):
                new_loc = tuple(wrap(np.array(row_col(goose[0])) + m))
                food_distances[goose_index, action_index] = min([DISTANCE_MATRIX[new_loc][f] for f in food_locs])

    # We add 1 as otherwise the agent would never consider moves other than those that allow it to eat once it was
    # adjacent to food
    return special.softmax(1. / ((food_distances + 1.) * ACTOR_SOFTMAX_MOD), axis=-1)


# Geese are evaluated based on the proportion of their length to the total length of all geese
# This value is then rescaled to account for geese that have already died
# Finally, the overall rankings and ranking estimates are rescaled to be from -1 to 1
def critic_func(state):
    geese = state[0]['observation']['geese']
    n_geese = len(geese)
    goose_lengths = np.array([len(goose) for goose in geese]).astype(np.float)
    dead_geese_mask = goose_lengths == 0
    agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
    agent_rankings_rescaled = agent_rankings / (n_geese - 1.)

    # Exclude goose tail locations when counting blocked moves since they will (usually) be gone the next step
    goose_locs = set()
    for goose in geese:
        if len(goose) > 1:
            goose_locs.update([row_col(segment) for segment in goose[:-1]])
    # All geese after the first step start with one blocked move - their last action
    goose_blocked_moves = np.ones_like(goose_lengths)
    for agent_idx, agent in enumerate(state):
        if goose_lengths[agent_idx] > 0:
            for a, m in MOVES:
                if a != Action[agent['action']].opposite().name:
                    new_loc = tuple(wrap(np.array(geese[agent_idx][0]) + m))
                    if new_loc in goose_locs:
                        goose_blocked_moves[agent_idx] += 1.
    softmax_mod = critic_softmax_mod_func(state[0]['observation']['step'])
    goose_lengths_norm = special.softmax(np.where(
        dead_geese_mask,
        float('-inf'),
        (goose_lengths * (1. - BLOCKED_MOVE_WEIGHT) - goose_blocked_moves * BLOCKED_MOVE_WEIGHT) * softmax_mod
    ))
    remaining_rewards = np.linspace(0., 1., n_geese)[dead_geese_mask.sum():]
    remaining_rewards_var = remaining_rewards.max() - remaining_rewards.min()
    goose_lengths_norm_rescaled = remaining_rewards.min() + goose_lengths_norm * remaining_rewards_var
    # This is a hacky solution - is there a more elegant way to do this?
    if dead_geese_mask.sum() == 0:
        goose_lengths_norm_rescaled *= 2.
    elif dead_geese_mask.sum() == 1:
        goose_lengths_norm_rescaled *= 1.2

    final_ranks = np.where(
        dead_geese_mask,
        agent_rankings_rescaled,
        goose_lengths_norm_rescaled
    )
    if not np.isclose(final_ranks.sum(), n_geese / 2.):
        raise RuntimeError(f'Final ranks should sum to {n_geese / 2.}\n'
                           f'Final ranks: {final_ranks}\n'
                           f'Dead geese mask: {dead_geese_mask}\n'
                           f'Returned rewards: {2. * final_ranks - 1.}')
    return 2. * final_ranks - 1.


def terminal_value_func(state):
    agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
    ranks_rescaled = 2. * agent_rankings / (len(state) - 1.) - 1.
    return ranks_rescaled


def actor_critic_func(state):
    return actor_func(state), critic_func(state)


# TODO: Tunable parameters
"""
ACTOR_SOFTMAX_MOD: [-inf, inf] How sensitive the actor is to it's distance from food.
    Larger values decrease the actor's propensity to greedily move towards food
BLOCKED_MOVE_WEIGHT: [0, 1] How much the critic should weight the number of blocked moves relative to goose length.
    Larger values increase the critic's desire to seek out open regions, but simultaneously decreases the critic's
    desire to eat and gain length
C_PUCT: [0, inf) How much the MCTS algorithm should prioritize the predictions of the actor
    relative to the predictions of the rollouts and critic
    Larger values increase the priority of the critic, whereas smaller values increase the priority of the actor
EXPECTED_END_STEP: Controls the time management of the agent
"""
ACTOR_SOFTMAX_MOD = 1.
BLOCKED_MOVE_WEIGHT = 0.25
C_PUCT = np.sqrt(2.)
EXPECTED_END_STEP = 200. - 10.

assert 0. <= BLOCKED_MOVE_WEIGHT <= 1.
assert C_PUCT >= 0.


def critic_softmax_mod_func(step: int) -> float:
    # return step / 200.
    return 1.


class Agent:
    def __init__(self, obs: Observation, conf: Configuration):
        self.index = obs.index

        self.search_tree = BasicMCTS(
            action_mask_func=action_mask_func,
            actor_critic_func=actor_critic_func,
            terminal_value_func=terminal_value_func,
            c_puct=C_PUCT,
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

    def __call__(self, obs: Observation, conf: Configuration):
        self.preprocess(obs, conf)
        env = make_from_state(obs, self.last_actions)
        # Remove excess nodes from dictionary to avoid memory explosion
        csr = env.canonical_string_repr(include_food=self.search_tree.include_food)
        for key in list(self.search_tree.nodes.keys()):
            if key.startswith(f'S: {obs.step - 1}') or (key.startswith(f'S: {obs.step}') and key != csr):
                del self.search_tree.nodes[key]
        # TODO: More intelligently allocate overage time
        max_time = 0.98 + (max(obs.remaining_overage_time - 1., 0.) / max(EXPECTED_END_STEP - obs.step, 1.))
        root_node = self.search_tree.run_mcts(
            env=env,
            n_iter=10000,
            max_time=max_time
        )
        # Greedily select best action
        policies = root_node.get_improved_policies(temp=1.)
        q_vals = root_node.q_vals
        my_best_action_idx = root_node.get_improved_actions(temp=0.)[self.index]
        selected_action = tuple(Action)[my_best_action_idx].name
        print(f'Step: {obs.step}', end=' ')
        print(f'Index: {self.index}', end=' ')
        print(f'My initial policy: {print_array_one_line(actor_func(env.state)[self.index])}', end=' ')
        print(f'My policy: {print_array_one_line(policies[self.index])}', end=' ')
        print(f'My Q-values: {print_array_one_line(q_vals[self.index])}', end=' ')
        print(f'Selected action: {selected_action}', end=' ')
        print(f'N-visits: {root_node.n_visits.sum(axis=1)[self.index]:.0f}', end=' ')
        print(f'Time allotted: {max_time:.2f}', end=' ')
        print(f'Remaining overage time: {obs.remaining_overage_time:.2f}', end=' ')
        print(f'All policies: {print_array_one_line(policies)}', end=' ')
        print(f'All Q-values: {print_array_one_line(q_vals)}', end=' ')
        print()
        return selected_action


AGENT = None


def call_agent(obs, conf):
    global AGENT

    obs = Observation(obs)
    conf = Configuration(conf)
    if AGENT is None:
        AGENT = Agent(obs, conf)

    return AGENT(obs, conf)
