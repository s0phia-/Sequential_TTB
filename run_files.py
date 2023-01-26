import numpy as np
from analysis.plot import gather_data, plot_gg, results_writer
from tetris.game import Tetris


def game_over(writer, cleared_lines, agent, run_id, env, x):
    """
    game is over. Save stats and reset
    """
    writer.writerow([x, cleared_lines, type(agent).__name__, run_id])
    env.reset()
    available_actions, actions_inc_terminal = env.get_after_states(include_terminal=True)
    return env, available_actions, actions_inc_terminal


def run_ttb_val(agent, env, num_episodes, writer, run_id):
    available_actions, _ = env.get_after_states(include_terminal=True)
    for x in range(num_episodes):
        done = False
        cleared_lines = 0
        while not done:
            # agent chooses an action, which is played
            action = agent.choose_action(available_actions)
            state_prime, reward, done, _ = env.step(action)
            # env.render()

            # env provides new actions for the agent to pick from, agent stores new knowledge and learns from it
            available_actions_prime, _ = env.get_after_states(include_terminal=True)
            agent.store_data(action, available_actions, reward)
            agent.learn()

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            available_actions = available_actions_prime

        env, available_actions, _ = game_over(writer, cleared_lines, agent, run_id, env, x)


def run_ttb_q(agent, env, num_episodes, writer, run_id):
    available_actions, _ = env.get_after_states(include_terminal=True)
    for x in range(num_episodes):
        done = False
        cleared_lines = 0
        while not done:
            # agent chooses an action, which is played
            action = agent.choose_action(available_actions)
            state_prime, reward, done, _ = env.step(action)
            # env.render()
            state = available_actions[action]

            # env provides new actions for the agent to pick from, agent stores new knowledge and learns from it
            available_actions_prime, _ = env.get_after_states(include_terminal=True)
            #agent.store_data(state, state_prime, reward, available_actions_prime, done)
            #agent.learn()

            # update vars for the next loop
            cleared_lines += reward
            print("cleared lines: " + str(cleared_lines))
            state = state_prime
            available_actions = available_actions_prime
        env, available_actions, _ = game_over(writer, cleared_lines, agent, run_id, env, x)
        state = env.get_current_state_features()


def run_simple(agent, env, num_episodes, writer, run_id):
    """
    for running really simple agents like all equal weights, that do no learning or storing data
    """
    for x in range(num_episodes):
        done = False
        cleared_lines = 0
        while not done:
            # env.print_current_tetromino()  ##
            after_state_features = env.get_after_states()
            i = agent.choose_action(after_state_features)
            # print(after_state_features[i])
            observation, reward, done, _ = env.step(i)
            # env.render()  ##
            cleared_lines += reward
        game_over(writer, cleared_lines, agent, run_id, env, x)


def run_ttb_rollouts(agent, env, num_episodes, writer, run_id, skip_learning=5):
    available_actions, _ = env.get_after_states(include_terminal=True)
    for x in range(num_episodes):
        done = False
        cleared_lines = 0
        i = 0
        while not done:
            # agent chooses an action, which is played
            actions, returns = env.perform_rollouts(available_actions, agent.choose_action)
            agent.store_data(actions, returns)
            if i % skip_learning == 0:
                agent.learn()

            action = agent.choose_action(available_actions)
            _, reward, done, _ = env.step(action)
            # env.render()

            # env provides new actions for the agent to pick from
            available_actions, _ = env.get_after_states(include_terminal=True)

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            i += 1
        env, available_actions, _ = game_over(writer, cleared_lines, agent, run_id, env, x)


def pool_run(agent_class, i, results_path):
    """

    :param agent_class:
    :param i:
    :param results_path:
    :return:
    """
    # write results to CSV files
    filepath = f'{results_path}/{agent_class.__name__}_{i}.csv'
    open_file, writer = results_writer(filepath, i)

    # Create a tetris env with directed features
    env = Tetris(10, 20, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()

    # create an agent (may need some info from the environment)
    state = env.get_current_state_features()
    agent = agent_class(env.num_features, state)

    # set up play loop
    num_episodes = 25

    run_ttb_rollouts(agent, env, num_episodes, writer, i)

    # stop writing to the csv file, plot results
    open_file.close()
