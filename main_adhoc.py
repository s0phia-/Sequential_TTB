from datetime import datetime
import os
import csv
import ast
from collections import defaultdict

from agents.q_learning import QLearning, TDLearning
from agents.ttb_exploit_q import ExploitQ
from run_files import run_q, run_simple, results_writer, run_v
from chasing_gridworld.chasing_gridworld import ChasingGridWorldAfterStates, ChasingGridWorld
from tetris.game import Tetris


if __name__ == "__main__":

    # # make folder to save results in
    # results_file = f'results/runtime_{datetime.now()}'
    # if not os.path.exists(results_file):
    #     os.makedirs(results_file)
    # env = ChasingGridWorld(5, 5)
    # agent = QLearning(env.num_features, actions=[[0, 1], [0, -1], [1, 0], [-1, 0]])
    # run_q(agent, env, num_episodes=10**2, results_path=results_file)


    #     for k, v0, v1, v2, v3 in csv.reader(f):
    #         qq_table[ast.literal_eval(k)] = [float(v0), float(v1), float(v2), float(v3)]
    # save_results = f'results/{results_file}/ttb_performance.csv'
    # env = ChasingGridWorldAfterStates(5, 5)
    # agent = ExploitQ(qq_table)
    # open_file, writer = results_writer(save_results)
    # run_simple(agent, env, 100, writer, 1)
    # open_file.close()
    #
    # # make folder to save results in
    # results_file = f'results/runtime_{datetime.now()}'
    # if not os.path.exists(results_file):
    #     os.makedirs(results_file)
    # env = Tetris(5, 5)
    # agent = TDLearning(env.num_features)
    # run_v(agent, env, num_episodes=10**6, results_path=results_file)

    # results_file = 'works'
    # path = f'results/{results_file}/v_table.csv'
    # vv_table = defaultdict(lambda: -100)
    # with open(path) as f:
    #     for k, v in csv.reader(f):
    #         vv_table[ast.literal_eval(k)] = ast.literal_eval(v)
    results_file = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_file):
        os.makedirs(results_file)
    env = Tetris(5, 5)
    agent = TDLearning(env.num_features)
    # agent.vv = vv_table
    agent.epsilon = 0
    run_v(agent, env, num_episodes=10**2, results_path=results_file)

