import copy
from kaggle_environments import make as kaggle_make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, histogram, translate
import numpy as np
from random import sample
from typing import *

from ..config import N_PLAYERS
from ..utils import STATE_TYPE


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

    def reset(self, num_agents: int = N_PLAYERS) -> STATE_TYPE:
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
        self.step_counter += 1

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
            if self.last_actions[index] == action.opposite() and self.step_counter > 1:
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
                    'geese': [[g for g in goose] for goose in self.geese],
                    'food': [f for f in self.food]
                })
            state_dict_list.append(dict_i)

        self.steps.append(state_dict_list)

    @property
    def state(self) -> STATE_TYPE:
        return self.steps[-1]

    @property
    def done(self) -> bool:
        n_geese_alive = len([True for goose in self.geese if len(goose) > 0])
        return n_geese_alive <= 1 or self.step_counter >= self.configuration.episode_steps - 1

    def get_statuses(self) -> List[str]:
        if self.done:
            return ['DONE' for _ in range(self.agent_count)]
        else:
            return ['ACTIVE' if len(goose) > 0 else 'DONE' for goose in self.geese]

    def lightweight_clone(self):
        cloned_env = LightweightEnv(self.configuration)

        cloned_env.agent_count = self.agent_count
        cloned_env.geese = [[g for g in goose] for goose in self.geese]
        cloned_env.food = [f for f in self.food]
        cloned_env.last_actions = [a for a in self.last_actions]
        cloned_env.step_counter = self.step_counter
        cloned_env.rewards = [r for r in self.rewards]
        cloned_env.steps = [None] * (len(self.steps) - 1)
        cloned_env.steps.append(copy.deepcopy(self.steps[-1]))

        return cloned_env

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


def make(environment: str = 'hungry_geese', debug: bool = False, **kwargs) -> LightweightEnv:
    assert environment == 'hungry_geese'
    config = Configuration(kaggle_make(environment, debug=debug, **kwargs).configuration)
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
