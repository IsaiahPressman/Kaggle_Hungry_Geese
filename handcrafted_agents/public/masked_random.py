import random
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation

last_action = None


def agent(obs_dict, config_dict):
    """Choose action randomly considering available actions"""
    global last_action

    # Check legal (available) actions
    legal_actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    if last_action is not None:
        legal_actions.remove(Action[last_action].opposite().name)

    last_action = random.choice(legal_actions)
    return last_action
