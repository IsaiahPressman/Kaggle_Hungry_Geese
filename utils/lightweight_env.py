import copy
from kaggle_environments import make as kaggle_make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, histogram, translate
from random import sample
from typing import *


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

    def reset(self, num_agents: int = 1) -> Tuple[List[Dict], Configuration]:
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

        return self.state, self.configuration

    def step(self, actions: List[str]):
        assert not self.done
        assert len(actions) == self.agent_count, f'Got {len(actions)} actions for {self.agent_count} agents'
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

        return self.state

    @property
    def state(self) -> List[Dict]:
        state_dict_list = []
        for i in range(self.agent_count):
            dict_i = {
                'action': self.last_actions[i].name,
                'reward': self.rewards[i],
                # 'info' is not used, and so is excluded
                'observation': {
                    # 'remainingOverageTime' is not computed, and so is excluded
                    'index': i
                }
                # 'status' is not used, and so is excluded
            }
            if i == 0:
                dict_i['observation'].update({
                    # 'remainingOverageTime' is not computed, and so is excluded
                    'step': self.step_counter,
                    'geese': self.geese,
                    'food': self.food
                })
            state_dict_list.append(dict_i)

        return state_dict_list

    @property
    def done(self) -> bool:
        n_geese_alive = len([True for goose in self.geese if len(goose) > 0])
        return n_geese_alive <= 1

    def clone(self):
        cloned_env = LightweightEnv(self.configuration)

        cloned_env.agent_count = self.agent_count
        cloned_env.geese = copy.deepcopy(self.geese)
        cloned_env.food = copy.copy(self.food)
        cloned_env.last_actions = copy.copy(self.last_actions)
        cloned_env.step_counter = self.step_counter
        cloned_env.rewards = self.rewards

        return cloned_env

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
                board[position] = index

        out = row_divider
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                out += column_divider + f" {board[(row * self.n_cols) + col]} "
            out += column_divider + "\n" + row_divider

        return out


def make(environment: str, debug: bool = False, **kwargs):
    assert environment == 'hungry_geese'
    config = kaggle_make(environment, debug=debug, **kwargs).configuration
    return LightweightEnv(config, debug=debug)
