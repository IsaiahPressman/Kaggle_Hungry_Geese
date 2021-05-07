from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments import make as kaggle_make
import numpy as np
try:
    import ujson as json
except ModuleNotFoundError:
    import json
from pathlib import Path
from typing import *

from .utils import read_ljson

CONFIG = {
    'actTimeout': 1,
    'columns': 11,
    'episodeSteps': 200,
    'hunger_rate': 40,
    'min_food': 2,
    'rows': 7,
    'runTimeout': 1200
}

INFO = {
    'EpisodeId': 0,
    'LiveVideoPath': None,
    'TeamNames': ['NA'] * 4
}


def render_ljson(file_path: Union[str, Path], **render_kwargs):
    steps_unformatted = read_ljson(file_path)

    steps = []
    for su in steps_unformatted:
        step = []
        for agent in su:
            agent_dict = {
                'action': agent['action'],
                'info': agent.get('info', {}),
                'observation': agent['observation'],
                'reward': agent['reward'],
                'status': agent['status']
            }
            agent_dict['observation'].setdefault('remainingOverageTime', 60.)
            step.append(agent_dict)
        steps.append(step)

    env = kaggle_make(
        'hungry_geese',
        configuration=CONFIG,
        steps=steps,
        info=INFO
    )

    action_names = [a.name for a in Action]
    final_actions = []
    for i, agent in enumerate(steps_unformatted[-1]):
        policy = np.array(agent.get('policy', [0.]))
        action = action_names[policy.argmax()]
        if (policy == policy.max()).sum() > 1:
            print(f'WARNING: Unknown action for goose {i}. Guessing {action}')
        final_actions.append(action)
    env.step(final_actions)

    return env.render(**render_kwargs)
