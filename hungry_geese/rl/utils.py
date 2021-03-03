import torch
from typing import *


def run_vs(agent_1, agent_2, n_agent_1s=2, n_agent_2s=2):
    # TODO
    assert False, 'Not yet implemented'


class EpsilonScheduler:
    def __init__(self,
                 start_vals: Union[float, torch.Tensor],
                 min_vals: Union[float, torch.Tensor],
                 train_steps_to_reach_min_vals: Union[int, float, torch.Tensor]):
        if type(start_vals) != torch.Tensor:
            start_vals = torch.tensor([start_vals], dtype=torch.float32)
        if type(min_vals) != torch.Tensor:
            min_vals = torch.tensor([min_vals], dtype=torch.float32)
        if type(train_steps_to_reach_min_vals) != torch.Tensor:
            train_steps_to_reach_min_vals = torch.tensor([train_steps_to_reach_min_vals], dtype=torch.float32)

        self.start_vals = start_vals
        self.min_vals = min_vals
        self.train_steps_to_reach_min_vals = train_steps_to_reach_min_vals

    def __call__(self, train_step):
        percent_to_min = train_step / self.train_steps_to_reach_min_vals
        decayed_epsilon = self.start_vals - percent_to_min * (self.start_vals - self.min_vals)
        return torch.maximum(decayed_epsilon, self.min_vals).unsqueeze(0)
