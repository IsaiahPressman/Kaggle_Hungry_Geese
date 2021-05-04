from filelock import FileLock
try:
    import ujson as json
except ModuleNotFoundError:
    import json
import torch.multiprocessing as mp
from pathlib import Path
import time
import torch
from torch import nn
from typing import *

from ...env.goose_env import ObsType
from ...env.torch_env import TorchEnv
from ...mcts.torch_mcts import TorchMCTS
from ...mcts.utils import torch_actor_critic_factory, torch_terminal_value_func
from hungry_geese.nns.models import FullConvActorCriticNetwork


def save_episodes_worker(
        dataset_dir: Path,
        save_episode_queue: mp.Queue,
        max_saved_episodes: int,
        start_idx: int = 0
) -> NoReturn:
    saved_episode_counter = start_idx
    raise NotImplementedError()


# TODO: Remember to send over policies at every step
# TODO: Remember to send over final_rankings on episode end

# Maybe share memory of original env directly, and then only perform search on env_copy.
# In this case, MAKE SURE not to step in original env until received go-ahead signal from save_episodes_worker that
# it has finished generating the obs_dicts
