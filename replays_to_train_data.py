import argparse
from copy import copy
try:
    import ujson as json
except ModuleNotFoundError:
    import json
import kaggle_environments
import numpy as np
import os
from pathlib import Path
import pandas as pd
from scipy import stats
import tqdm
from typing import *

from hungry_geese.utils import STATE_TYPE, read_json


def process_replay_file(replay_dict: Dict, index_to_mmr: pd.Series) -> List[STATE_TYPE]:
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
            agent['next_action'] = env.steps[step_idx + 1][agent_idx]['action']
            agent['final_rank'] = agent_rankings[agent_idx]
            agent['mmr'] = index_to_mmr[agent_idx].item()

    return env.steps[:-1]


def batch_split_replay_files(
        epagents_df: pd.DataFrame,
        replay_paths_to_save: List[Path],
        save_dir: Path,
        force: bool,
        delete: bool
) -> NoReturn:
    all_replay_paths_to_save = copy(replay_paths_to_save)
    saved_replay_names = []
    if save_dir.exists():
        assert save_dir.is_dir()
        if (save_dir / 'all_processed_episodes.txt').exists() and not force:
            already_processed = set()
            if (save_dir / 'all_processed_episodes.txt').is_file():
                with open(save_dir / 'all_processed_episodes.txt', 'r') as f:
                    already_processed.update([replay_name.rstrip() for replay_name in f.readlines()])
                replay_paths_to_save = [rp for rp in replay_paths_to_save if rp.stem not in already_processed]
            if (save_dir / 'all_saved_episodes.txt').is_file():
                with open(save_dir / 'all_saved_episodes.txt', 'r') as f:
                    saved_replay_names.extend([replay_name.rstrip() for replay_name in f.readlines()])
    else:
        save_dir.mkdir()

    step_counter = 0
    print(f'Processing {len(replay_paths_to_save)} replays and saving output to {save_dir.absolute()}')
    for rp in tqdm.tqdm(copy(replay_paths_to_save)):
        try:
            episode = process_replay_file(
                read_json(rp),
                epagents_df[epagents_df.EpisodeId == int(rp.stem)].set_index('Index').LatestScore
            )
            save_file_name = save_dir / (rp.stem + '.ljson')
            if not save_file_name.exists() or force:
                with open(save_file_name, 'w') as f:
                    f.writelines([json.dumps(step) + '\n' for step in episode])
                step_counter += len(episode)
                saved_replay_names.append(rp.stem)
            else:
                raise RuntimeError(f'Replay already exists and force is False: {(save_dir / rp.name)}')
        except (kaggle_environments.errors.InvalidArgument, RuntimeError) as e:
            print(f'Unable to save replay {rp.name}:')
            replay_paths_to_save.remove(rp)
            print(e)
        except ValueError:
            print(f'Unable to save empty or malformed replay {rp.name} - deleting')
            all_replay_paths_to_save.remove(rp)
            os.remove(rp)

    all_replay_names = set([rp.stem for rp in all_replay_paths_to_save])
    saved_replay_names = set(saved_replay_names)
    if delete:
        found_episodes = list(save_dir.glob('*.ljson'))
        for ep_path in found_episodes:
            if ep_path.stem not in all_replay_names:
                os.remove(ep_path)
                if ep_path.stem in saved_replay_names:
                    saved_replay_names.remove(ep_path.stem)

    all_replay_names = sorted(list(all_replay_names), key=lambda rn: int(rn))
    saved_replay_names = sorted(list(saved_replay_names), key=lambda rn: int(rn))
    with open(save_dir / 'all_processed_episodes.txt', 'w') as f:
        f.writelines([f'{rn}\n' for rn in all_replay_names])
    with open(save_dir / 'all_saved_episodes.txt', 'w') as f:
        f.writelines([f'{rn}\n' for rn in saved_replay_names])
    print(f'Successfully saved {step_counter:,} steps from {len(replay_paths_to_save)} replays.')
    print(f'{len(saved_replay_names)} out of {len(all_replay_names)} replays saved in total.')


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    # Load Episodes and EpisodeAgents, and filter for hungry-geese competition
    episodes_df = pd.read_csv(metadata_path / 'Episodes.csv')
    episodes_df = episodes_df[episodes_df.CompetitionId == 25401]

    epagents_df = pd.read_csv(metadata_path / 'EpisodeAgents.csv')
    epagents_df = epagents_df[epagents_df.EpisodeId.isin(episodes_df.Id)]
    epagents_df.fillna(0, inplace=True)
    epagents_df = epagents_df.sort_values(by=['Id'], ascending=False)

    latest_scores_df = epagents_df.loc[epagents_df.groupby('SubmissionId').EpisodeId.idxmax(), :].sort_values(
        by=['UpdatedScore'])
    latest_scores_df['LatestScore'] = latest_scores_df.UpdatedScore
    latest_scores_df = latest_scores_df[['SubmissionId', 'LatestScore']]
    epagents_df = epagents_df.merge(latest_scores_df, left_on='SubmissionId', right_on='SubmissionId',
                                    how='outer').sort_values(by=['LatestScore'])

    return epagents_df


def select_episodes(epagents_df: pd.DataFrame, replay_paths: List[Path], threshold: float) -> List[Path]:
    episode_min_scores = epagents_df.groupby('EpisodeId').LatestScore.min()
    ep_to_score = episode_min_scores[episode_min_scores >= threshold].to_dict()

    return [rp for rp in replay_paths if int(rp.stem) in ep_to_score.keys()]


def main() -> NoReturn:
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
        help='A list of JSON replay file paths, or a relative glob pattern to match to find all replay paths'
    )
    parser.add_argument(
        '-d',
        '--delete',
        action='store_true',
        help='Including this option will delete episodes that no longer qualify for the given threshold'
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Including this option will overwrite existing saved episodes'
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
    if len(args.replay_paths) == 1:
        args.replay_paths = list(Path('.').glob(str(args.replay_paths[0])))

    epagents_df = load_metadata(args.metadata_path)
    args.save_dir.mkdir(exist_ok=True)
    if args.threshold is not None:
        selected_replay_paths = select_episodes(epagents_df, args.replay_paths, args.threshold)
    else:
        selected_replay_paths = args.replay_paths
    batch_split_replay_files(epagents_df, selected_replay_paths, args.save_dir, args.force, args.delete)


if __name__ == '__main__':
    main()
