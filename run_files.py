from tetris.game import Tetris
import csv


def results_writer(results_file):
    """
    create file ready to write to
    :param results_file: filepath to where results should be saved
    :return:
    """
    f = open(results_file, 'w')
    writer = csv.writer(f)
    writer.writerow(['episode_number', 'step', 'return', 'agent', 'run_id'])
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

            # env provides new actions for the agent to pick from, agent stores new knowledge and learns from it
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

            # env provides new actions for the agent to pick from, agent stores new knowledge and learns from it
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


def run_simple(agent, env, num_episodes, writer, run_id):
    """
    for running really simple agents like all equal weights, that do no learning or storing data
    """
    for ep in range(num_episodes):
        done = False
        cleared_lines = 0
        step = 0
        while not done:
            # env.print_current_tetromino()  ##
            after_state_features, _ = env.get_after_states()
            i = agent.choose_action(after_state_features)
            # print(after_state_features[i])
            observation, reward, done, _ = env.step(i)
            step += 1
            # env.render()  ##
            cleared_lines += reward
            writer.writerow([ep, step, cleared_lines, type(agent).__name__, run_id])
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

            # env provides new actions for the agent to pick from
            available_actions, _ = env.get_after_states(include_terminal=True)

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            i += 1
            writer.writerow([ep, step, cleared_lines, type(agent).__name__, run_id])
        env.reset()


def run_ttb_q_seq(agent, env, num_episodes, writer, run_id, skip_learning=1):

    for ep in range(num_episodes):
        available_actions, _ = env.get_after_states(include_terminal=True)
        done = False
        cleared_lines = 0
        step = 0

        while not done:
            state = env.get_current_state_features()
            if step % skip_learning == 0:
                # agent chooses an action, which is played
                actions, rewards = env.perform_rollouts(available_actions, agent.choose_action, length=1, n=3)
                agent.learn(state, actions, rewards)

            action = agent.choose_action(available_actions)
            _, reward, done, _ = env.step(action)
            step += 1
            # env.render()

            # env provides new actions for the agent to pick from
            available_actions, _ = env.get_after_states(include_terminal=True)

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            writer.writerow([ep, step, cleared_lines, type(agent).__name__, run_id])
            if cleared_lines >= 1000:
                break
        env.reset()
        available_actions, actions_inc_terminal = env.get_after_states(include_terminal=True)


def pool_run(agent_class, i, results_path, num_episodes, tetris_cols, tetris_rows, play_loop):
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
    env = Tetris(tetris_rows, tetris_cols, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()

    # create an agent (may need some info from the environment)
    state = env.get_current_state_features()
    agent = agent_class(env.num_features, state)

    # set up play loop
    play_loop(agent, env, num_episodes, writer, i)

    # stop writing to the csv file, plot results
    open_file.close()
