from tetris.game import Tetris
from chasing_gridworld.chasing_gridworld import ChasingGridWorld, ChasingGridWorldAfterStates
from chasing_gridworld.simpler_gridworld import SimpleGridworldAfterStates
import csv
import numpy as np


def results_writer(results_file):
    """
    create file ready to write to
    :param results_file: filepath to where results should be saved
    :return:
    """
    f = open(results_file, 'w')
    writer = csv.writer(f)
    writer.writerow(['step', 'return'])
    return f, writer


def run_ttb_val(agent, env, num_episodes, writer, run_id):
    for ep in range(num_episodes):
        available_actions, _ = env.get_after_states(include_terminal=True)
        done = False
        cleared_lines = 0
        step = 0
        while not done:
            # agent chooses an action, which is played
            action = agent.choose_action(available_actions)
            state_prime, reward, done, _ = env.step(action)
            step += 1
            # env.render()

            # env provides new after_states for the agent to pick from, agent stores new knowledge and learns from it
            available_actions_prime, _ = env.get_after_states(include_terminal=True)
            agent.store_data(action, available_actions, reward)
            agent.learn()

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            available_actions = available_actions_prime
            writer.writerow([ep, step, cleared_lines, type(agent).__name__, run_id])
        env.reset()


def run_ttb_q(agent, env, num_episodes, writer, run_id):
    for ep in range(num_episodes):
        available_actions, _ = env.get_after_states(include_terminal=True)
        done = False
        cleared_lines = 0
        step = 0
        while not done:
            # agent chooses an action, which is played
            action = agent.choose_action(available_actions)
            state_prime, reward, done, _ = env.step(action)
            step += 1
            # env.render()
            state = available_actions[action]

            # env provides new after_states for the agent to pick from, agent stores new knowledge and learns from it
            available_actions_prime, _ = env.get_after_states(include_terminal=True)
            agent.store_data(state, state_prime, reward, available_actions_prime, done)
            agent.learn()

            # update vars for the next loop
            cleared_lines += reward
            print("cleared lines: " + str(cleared_lines))
            state = state_prime
            available_actions = available_actions_prime
            writer.writerow([ep, step, cleared_lines, type(agent).__name__, run_id])
        env.reset()


def run_simple(agent, env, num_episodes, writer):
    """
    for running really simple agents like all equal weights, that do no learning or storing data
    """
    for ep in range(num_episodes):
        done = False
        total_cleared_lines = 0
        step = 0
        state = env.get_state()
        while not done:
            # env.print_current_tetromino()  ##
            afterstates, _ = env.get_after_states()
            action = agent.choose_action(state, afterstates)
            state_, reward, done, cleared_lines = env.step(action)
            step += 1
            # agent.learn(state, action, reward, afterstates)
            # env.render()  ##
            total_cleared_lines += cleared_lines
            writer.writerow([ep, step, total_cleared_lines])
            state = state_
            if step > 1000:
                break
        print(cleared_lines)
        env.reset()


def run_ttb_rollouts(agent, env, num_episodes, writer, run_id, skip_learning=1):
    for ep in range(num_episodes):
        available_actions, _ = env.get_after_states(include_terminal=True)
        done = False
        cleared_lines = 0
        i = 0
        step = 0
        while not done:
            if i % skip_learning == 0:
                # agent chooses an action, which is played
                actions, returns = env.perform_rollouts(available_actions, agent.choose_action)
                agent.store_data(actions, returns)
                agent.learn()

            action = agent.choose_action(available_actions)
            _, reward, done, _ = env.step(action)
            step += 1
            # env.render()

            # env provides new after_states for the agent to pick from
            available_actions, _ = env.get_after_states(include_terminal=True)

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            i += 1
            writer.writerow([ep, step, cleared_lines, type(agent).__name__, run_id])
        env.reset()


def run_ttb_q_seq(agent, env, num_episodes, writer, skip_learning=1):
    """
    Hello
    """
    step = 0

    for ep in range(num_episodes):

        done = False
        total_return = 0
        total_lines_cleared = 0
        ep_step = 0

        while not done:
            available_actions, _ = env.get_after_states(include_terminal=True)
            state = env.get_state()

            # if step % skip_learning == 0:
            #     actions, rewards = env.perform_rollouts(available_actions, agent.choose_action, length=1, n=1)
            #     agent.learn(state, actions, rewards)

            action = agent.choose_action(state, available_actions)

            # action_afterstate = available_actions[agent.choose_action(available_actions)]  # TODO: this is janky
            # action = env.get_action_from_afterstate(action_afterstate)

            _, reward, done, lines_cleared = env.step(action)
            ep_step += 1
            step += 1
            print(lines_cleared)
            env.render()

            #  agent.learn(state, action, reward, available_actions)

            if step % 5 == 0:
                get_performance(agent, 5, writer, step)

            total_return += reward
            total_lines_cleared += lines_cleared

            if done:
                print(total_lines_cleared)

            #  writer.writerow([ep, step, total_lines_cleared])

            if ep_step > 1000:
                print(total_lines_cleared)
                break
        env.reset()


def run_q(agent, env, num_episodes, results_path):

    def save_vv_table(learned_v_table, results_path):
        filepath = f'{results_path}/v_table.csv'
        with open(filepath, 'w') as f:
            writer2 = csv.writer(f)
            for k, v in learned_v_table.items():
                writer2.writerow([k, v])

    # save episode, step and return
    filepath = f'{results_path}/q_learning_returns.csv'
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['episode_number', 'step', 'return'])

        # q learning loop
        for ep in range(num_episodes):
            done = False
            step = 0
            total_return = 0
            while not done:
                state = env.get_state()
                action = agent.choose_action(state)
                state_, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, state_)
                step += 1
                total_return += reward
                writer.writerow([ep, step, total_return])
                if step > 1000:
                    break
            env.reset()

    # save Q table
    learned_q_table = agent.qq
    filepath = f'{results_path}/q_table.csv'
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        for k, v in learned_q_table.items():
            writer.writerow([k, v[0], v[1], v[2], v[3]])


def run_v(agent, env, num_episodes, writer):

    for ep in range(num_episodes):
        done = False
        step = 0
        total_lines_cleared = 0
        while not done:
            state = env.get_state()
            afterstates, _ = env.get_after_states()
            afterstate_ix = agent.choose_action(state, afterstates)
            state_, reward, done, lines_cleared = env.step(afterstate_ix)
            step += 1
            total_lines_cleared += lines_cleared
            agent.learn(state, reward, state_, done)
            if step % 5 == 0:
                pass
                #  get_performance(agent, 5, writer, step)
            if done:
                print(total_lines_cleared)
            if step > 1000:
                print(total_lines_cleared)
                break
        env.reset()


def get_performance(agent, num_episodes, writer, step_to_report):
    """
    judge performance without learning.
    """
    env = Tetris(8, 8, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()
    cleared_lines_per_ep = np.zeros(num_episodes)
    for ep in range(num_episodes):
        done = False
        total_cleared_lines = 0
        step = 0
        state = env.get_state()
        while not done:
            afterstates, _ = env.get_after_states()
            action = agent.choose_action(state, afterstates)
            state, reward, done, cleared_lines = env.step(action)
            step += 1
            total_cleared_lines += cleared_lines
            if step > 1000:
                break
        env.reset()
        cleared_lines_per_ep[ep] = total_cleared_lines
    av_cleared_lines = np.mean(cleared_lines_per_ep)
    print(step_to_report, av_cleared_lines)
    writer.writerow([step_to_report, av_cleared_lines])


def pool_run(agent_class, i, results_path, num_episodes, rows, cols, play_loop):
    """
    Need a separate run method for multi-processing. Runs a comparison of agents

    :param agent_class: class of agent being run. See full list imported in main
    :param i: agent index, for when multiple agents' performance is being averaged
    :param results_path: path to save results to
    :param num_episodes: number of episodes to run an agent for
    """
    # write results to CSV files
    filepath = f'{results_path}/{agent_class.__name__}_{i}.csv'
    open_file, writer = results_writer(filepath)

    # Create a tetris env with directed features
    env = Tetris(rows, cols, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()

    # create an agent (may need some info from the environment)
    state = env.get_state()
    agent = agent_class(num_features=env.num_features, current_state=state)

    # set up play loop
    play_loop(agent, env, num_episodes, writer)

    # stop writing to the csv file, plot results
    open_file.close()
