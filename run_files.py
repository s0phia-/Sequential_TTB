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

            # env provides new actions for the agent to pick from, agent stores new knowledge and learns from it
            available_actions_prime, _ = env.get_after_states(include_terminal=True)
            agent.store_data(state, state_prime, reward, available_actions_prime, done)
            agent.learn()

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
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


def run_ttb_rollouts(agent, env, num_episodes, writer, run_id):
    available_actions, _ = env.get_after_states(include_terminal=True)
    for x in range(num_episodes):
        done = False
        cleared_lines = 0
        while not done:
            # agent chooses an action, which is played
            env.perform_rollouts(available_actions)
            agent.store_data()
            agent.learn()

            action = agent.choose_action(available_actions)
            _, reward, done, _ = env.step(action)
            # env.render()

            # env provides new actions for the agent to pick from
            available_actions, _ = env.get_after_states(include_terminal=True)

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
        env, available_actions, _ = game_over(writer, cleared_lines, agent, run_id, env, x)
