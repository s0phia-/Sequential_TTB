from datetime import datetime
import os

from agents.q_learning import QLearning
from run_files import run_q
from chasing_gridworld.chasing_gridworld import ChasingGridWorld

if __name__ == "__main__":

    # make folder to save results in
    results_file = f'results/runtime_{datetime.now()}'
    if not os.path.exists(results_file):
        os.makedirs(results_file)
    env = ChasingGridWorld(10, 10)
    agent = QLearning(env.num_features, actions=[[0, 1], [0, -1], [1, 0], [-1, 0]])
    run_q(agent, env, num_episodes=10**7, results_path=results_file)
