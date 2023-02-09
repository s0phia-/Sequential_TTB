from agents.ttb_q import TtbQValues
from agents.pure_ew import EqualWeights
from agents.hand_crafted import HandCrafted
from agents.ttbvalidities import TtbValidities
from agents.ttb_rollout_validities import TtbRollVal, TtbRRollCondVal
from agents.RandomTTB import RandomTTB
from agents.ttb_rollouts import TtbRolloutsCorrelation, TtbRolloutsSummingScore
from agents.ttb_sequential import TTBFixedWeights
from agents.state_dependent_ttb import StateDependent
from run_files import run_ttb_val, run_ttb_q, run_ttb_rollouts, run_simple, pool_run, run_ttb_q_seq

from datetime import datetime
import os
import multiprocessing as mp


if __name__ == "__main__":

    # make folder to save results in
    results_file = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_file):
        os.makedirs(results_file)

    # create multiprocessing pools
    pool = mp.Pool(mp.cpu_count())

    # parameters
    agents = [[StateDependent, run_ttb_q_seq]]  # agents to try

    number_of_agents = 15  # number of agents to average performance over
    num_episodes = 10000
    tetris_cols = 5
    tetris_rows = 5

    all_run_args = [[agent_name, agent_i, results_file, num_episodes, tetris_cols, tetris_rows, play_loop]
                    for agent_name, play_loop in agents
                    for agent_i in range(number_of_agents)]

    pool.starmap(pool_run, all_run_args)
