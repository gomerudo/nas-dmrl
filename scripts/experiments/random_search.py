import os
from datetime import datetime
from random import seed
from random import random
from random import randint
import argparse

import gym
import nasgym
import pandas as pd

def do_random_search(random_seed=1024, log_dir="workspace_rs_mdn10", 
                     ntimesteps=8000):
    env = gym.make("NAS_cifar10-v1")

    seed(random_seed)

    episode_log_dir = "{dir}/episode_logs".format(
        dir=log_dir
    )
    os.makedirs(episode_log_dir, exist_ok=True)
    episode_log_path = "{dir}/{name}.csv".format(
        dir=episode_log_dir,
        name="episodes_results"
    )
    episode_df = None

    for _ in range(ntimesteps):
        action = randint(0, env.action_space.n)
        obs, rew, done, info_dict = env.step(action)

        if episode_df is None:
            headers = info_dict.keys()
            episode_df = pd.DataFrame(columns=headers)

        # TODO: Check if this works
        episode_df = episode_df.append(
            info_dict, ignore_index=True
        )

        if done:
            # Every time we are done, we will save the csv's
            if hasattr(env, 'save_db_experiments'):
                print("Saving database of experiments")
                env.save_db_experiments()
            if episode_df is not None:
                outfile = open(episode_log_path, 'a')
                print("Saving episode logs")
                episode_df.to_csv(outfile)
                outfile.close()
                episode_df = None

            env.reset()


    # Save at the end too
    if hasattr(env, 'save_db_experiments'):
        print("Saving database of experiments")
        env.save_db_experiments()

    if episode_df is not None:
        outfile = open(episode_log_path, 'a')
        print("Saving episode logs")
        episode_df.to_csv(outfile)
        outfile.close()
        episode_df = None

if __name__ == '__main__':
    # Define the arguments

    parser = argparse.ArgumentParser(
        description='Run a random search for NAS'
    )
    parser.add_argument('--log_dir', action="store", dest="log_dir")
    parser.add_argument('--ntimesteps', action="store", dest="ntimesteps")
    parser.add_argument('--random_seed', action="store", dest="random_seed")

    # Obtain the arguments of interest
    cmd_args = parser.parse_args()
    log_dir = cmd_args.log_dir if cmd_args.log_dir is not None else "workspace_rs_mdn10"
    ntimesteps = cmd_args.ntimesteps if cmd_args.ntimesteps is not None else 8000
    random_seed = cmd_args.random_seed if cmd_args.random_seed is not None else 1024

    print("Log dir is", log_dir)
    print("Ntimesteps", ntimesteps)
    print("random_seed", random_seed)

    do_random_search(random_seed, log_dir, ntimesteps)
