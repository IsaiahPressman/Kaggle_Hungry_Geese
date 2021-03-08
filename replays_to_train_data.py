import argparse
from copy import copy
import json
import kaggle_environments
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
import tqdm
from typing import *

from hungry_geese.utils import read_json


def split_replay_file(replay_dict: Dict) -> List[List[Dict]]:
    env = kaggle_environments.make(
        'hungry_geese',
        configuration=replay_dict['configuration'],
        steps=replay_dict['steps'],
        info=replay_dict['info']
    )

    game_score = np.array([agent['reward'] for agent in env.steps[-1]])
    agent_rankings = stats.rankdata(game_score, method='average') - 1.
    for step_idx, step in enumerate(env.steps[:-1]):
        for agent_idx, agent in enumerate(step):
            # agent['action'] is renamed to 'last_action' to avoid confusion
            agent['last_action'] = agent['action']
            del agent['action']
            agent['next_action'] = env.steps[step_idx + 1][agent_idx]['action']
            agent['final_rank'] = agent_rankings[agent_idx]

    return env.steps[:-1]


def batch_split_replay_files(replay_paths: List[Path], save_dir: Path, force: bool) -> NoReturn:
    all_replay_paths = copy(replay_paths)
    saved_replay_names = []
    if save_dir.exists():
        assert save_dir.is_dir()
        if (save_dir / 'all_processed_episodes.txt').exists() and not force:
            already_processed = set()
            if (save_dir / 'all_processed_episodes.txt').is_file():
                with open(save_dir / 'all_processed_episodes.txt', 'r') as f:
                    already_processed.update([replay_name.rstrip() for replay_name in f.readlines()])
                replay_paths = [rp for rp in replay_paths if rp.stem not in already_processed]
            if (save_dir / 'all_saved_episodes.txt').is_file():
                with open(save_dir / 'all_saved_episodes.txt', 'r') as f:
                    saved_replay_names.extend([replay_name.rstrip() for replay_name in f.readlines()])
    else:
        save_dir.mkdir()

    file_counter = 0
    print(f'Processing {len(replay_paths)} replays and saving output to {save_dir.absolute()}')
    for rp in tqdm.tqdm(copy(replay_paths)):
        try:
            steps = split_replay_file(read_json(rp))
            if not (save_dir / rp.stem).exists() or force:
                (save_dir / rp.stem).mkdir(exist_ok=True)
                for i, step in enumerate(steps):
                    with open(save_dir / rp.stem / f'{i}.json', 'w') as f:
                        json.dump(step, f)
                        file_counter += 1
                    saved_replay_names.append(rp.stem)
            else:
                raise RuntimeError(f'Folder already exists and force is False: {(save_dir / rp.stem)}')
        except (kaggle_environments.errors.InvalidArgument, RuntimeError) as e:
            print(f'Unable save replay {rp.name}:')
            replay_paths.remove(rp)
            print(e)

    all_replay_names = list(set([rp.stem for rp in all_replay_paths]))
    saved_replay_names = list(set(saved_replay_names))
    all_replay_names.sort(key=lambda rn: int(rn))
    saved_replay_names.sort(key=lambda rn: int(rn))
    with open(save_dir / 'all_processed_episodes.txt', 'w') as f:
        f.writelines([f'{rn}\n' for rn in all_replay_names])
    with open(save_dir / 'all_saved_episodes.txt', 'w') as f:
        f.writelines([f'{rn}\n' for rn in saved_replay_names])
    print(f'Successfully saved {file_counter:,} steps from {len(replay_paths)} replays.')
    print(f'{len(saved_replay_names)} out of {len(all_replay_names)} replays saved in total.')


def select_episodes(replay_paths: List[Path], metadata_path: Path, threshold: float) -> List[Path]:
    # Load Episodes and EpisodeAgents, and filter for hungry-geese competition
    episodes_df = pd.read_csv(metadata_path / 'Episodes.csv')
    epagents_df = pd.read_csv(metadata_path / 'EpisodeAgents.csv')
    episodes_df = episodes_df[episodes_df.CompetitionId == 25401]
    epagents_df = epagents_df[epagents_df.EpisodeId.isin(episodes_df.Id)]

    epagents_df.fillna(0, inplace=True)
    epagents_df = epagents_df.sort_values(by=['Id'], ascending=False)

    latest_scores_df = epagents_df.loc[epagents_df.groupby('SubmissionId').EpisodeId.idxmax(), :].sort_values(
        by=['UpdatedScore'])
    latest_scores_df['LatestScore'] = latest_scores_df.UpdatedScore
    latest_scores_df = latest_scores_df[['SubmissionId', 'LatestScore']]
    epagents_df = epagents_df.merge(latest_scores_df, left_on='SubmissionId', right_on='SubmissionId',
                                    how='outer').sort_values(by=['LatestScore'])

    episode_min_scores = epagents_df.groupby('EpisodeId').LatestScore.min()
    ep_to_score = episode_min_scores[episode_min_scores >= threshold].to_dict()

    return [rp for rp in replay_paths if int(rp.stem) in ep_to_score.keys()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process a list of JSON replay files and creates a new file per step.'
    )
    parser.add_argument(
        'save_dir',
        type=Path,
        help='Where to save the .pt output files'
    )
    parser.add_argument(
        'replay_paths',
        nargs='+',
        type=Path,
        help='A list of JSON replay file paths'
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Including this option will overwrite existing folders'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=None,
        help='The minimum ELO threshold of agents to include an episode. Leave empty to process all episodes'
    )
    parser.add_argument(
        '-m',
        '--metadata_path',
        type=Path,
        default=Path('episode_scraping/metadata'),
        help='The path to directory containing the EpisodeAgents and Episodes .csv files. '
             'Default: episode_scraping/metadata'
    )
    args = parser.parse_args()
    if args.threshold is not None:
        selected_replay_paths = select_episodes(args.replay_paths, args.metadata_path, args.threshold)
    else:
        selected_replay_paths = args.replay_paths
    batch_split_replay_files(selected_replay_paths, args.save_dir, args.force)
