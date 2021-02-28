import argparse
import contextlib
import io
import multiprocessing
import os
import numpy as np
from pathlib import Path
from scipy import stats
import time
import tqdm

with contextlib.redirect_stdout(io.StringIO()):
    # Silence gfootball import error
    from kaggle_environments import make

os.environ['OMP_NUM_THREADS'] = '1'
SEP = ', '
debug = None
save_dir = None


def init_worker(debug_mode, save_dir_):
    global debug, save_dir
    debug = debug_mode
    save_dir = save_dir_


def get_game_result(agents):
    env = make('hungry_geese', debug=debug)
    env.run(agents)
    rendered_html = env.render(mode='html')
    ep_length = env.steps[-1][0]['observation']['step']
    replay_n = 0
    while list(save_dir.glob(f'replay_{replay_n}*')):
        replay_n += 1
    with open(save_dir / f'replay_{replay_n}_{ep_length}_steps.html', 'w') as f:
        f.write(rendered_html)
    return [agent['reward'] if agent['reward'] is not None else -100. for agent in env.steps[-1]], len(env.steps)


if __name__ == '__main__':
    start_time = time.time()

    default_save_dir = Path(Path(__file__).parent) / 'replays'
    parser = argparse.ArgumentParser(description='Compare multiple agents in an asynchronous multi-game match.')
    parser.add_argument(
        'agent_paths',
        nargs='+',
        type=str,
        help='The paths to the agents to be compared. 1, 2, or 4 paths may be provided'
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='Including this flag runs the games in debug mode'
    )
    parser.add_argument(
        '-g',
        '--n_games',
        type=int,
        default=(multiprocessing.cpu_count()-1)*2,
        help=f'The number of games to play. Default: {(multiprocessing.cpu_count()-1)*2}'
    )
    parser.add_argument(
        '-s',
        '--save_dir',
        type=Path,
        default=default_save_dir,
        help=f'Where to save replays. Default: {default_save_dir}'
    )
    parser.add_argument(
        '-w',
        '--n_workers',
        type=int,
        default=multiprocessing.cpu_count()-1,
        help=f'The number of worker processes to use. Default: {multiprocessing.cpu_count()-1}'
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    save_dir.mkdir(exist_ok=True)
    subfolder_idx = 0
    subfolder = save_dir / str(subfolder_idx)
    while subfolder.is_dir() and os.listdir(subfolder):
        subfolder_idx += 1
        subfolder = save_dir / str(subfolder_idx)
    save_dir = save_dir / str(subfolder_idx)
    save_dir.mkdir(exist_ok=True)

    if len(args.agent_paths) == 1:
        agent_paths = args.agent_paths * 4
    elif len(args.agent_paths) == 2:
        agent_paths = [args.agent_paths[0], args.agent_paths[0],
                       args.agent_paths[1], args.agent_paths[1]]
    elif len(args.agent_paths) == 4:
        agent_paths = args.agent_paths
    else:
        raise ValueError(f'1, 2, or 4 agent paths must be provided, got {len(args.agent_paths)}')
    vs_message = ' -vs- '.join([Path(ap).stem for ap in agent_paths])
    print(vs_message)
    
    if args.n_workers == 1:
        results_and_game_lengths = []
        init_worker(args.debug, args.save_dir)
        for i in tqdm.trange(args.n_games):
            results_and_game_lengths.append(get_game_result(agent_paths))
    else:
        agent_paths_broadcasted = []
        for i in range(args.n_games):
            agent_paths_broadcasted.append(agent_paths)
        with multiprocessing.Pool(
                processes=args.n_workers,
                initializer=init_worker,
                initargs=(args.debug, save_dir)
        ) as pool:
            results_and_game_lengths = list(tqdm.tqdm(pool.imap(get_game_result, agent_paths_broadcasted),
                                                      total=args.n_games))

    all_results = []
    all_game_lengths = []
    for i, (result, game_length) in enumerate(results_and_game_lengths):
        game_scores = np.array(result)
        agent_rankings = stats.rankdata(game_scores, method='average') - 1.
        ranks_rescaled = 2. * agent_rankings / (4. - 1.) - 1.
        all_results.append(ranks_rescaled)
        all_game_lengths.append(game_length)
        print(f'Round {i+1}: {SEP.join([f"{r:.2f}" for r in ranks_rescaled])}')
    all_results = np.vstack(all_results)

    print(vs_message)
    mean_scores_out = 'Mean scores: '
    for score in all_results.mean(axis=0):
        mean_scores_out += f'{score:.2f}{SEP}'
    mean_scores_out = mean_scores_out[:-len(SEP)]

    std_scores_out = 'Score standard deviations: '
    for score in all_results.std(axis=0):
        std_scores_out += f'{score:.2f}{SEP}'
    std_scores_out = std_scores_out[:-len(SEP)]

    print(mean_scores_out)
    print(std_scores_out)
    print(f'Mean game length: {np.mean(all_game_lengths):.2f}')
    print(f'Finished in {int(time.time() - start_time)} seconds')
