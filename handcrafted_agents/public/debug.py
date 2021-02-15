from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col


def agent(obs_dict, config_dict):
    """This agent always moves NORTH, and is used for debugging"""
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    print(obs_dict)
    print(config_dict)
    print(observation)
    print(configuration)
    print()
    
    return Action.NORTH.name
