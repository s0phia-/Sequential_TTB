from tetris.game import Tetris
from agents.ttb_q import Ttb_Q_values
from agents.pure_ew import EqualWeights
from agents.hand_crafted import HandCrafted
from agents.ttb_validities import Ttb_validities
from analysis.plot import gather_data, plot_gg, results_writer


if __name__ == "__main__":

    # write results to CSV files
    results_file = 'results/result1.csv'
    open_file, writer, run_id = results_writer(results_file)

    # Create a tetris env with directed features
    env = Tetris(10, 10, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()

    # create an agent (may need some info from the environment)
    state = env.get_current_state_features()
    agent = Ttb_validities(env.num_features, state)

    # set up play loop
    done = False
    cleared_lines = 0
    num_episodes = 100
    available_actions, actions_inc_terminal = env.get_after_states(include_terminal=True)
    for x in range(num_episodes):
        while not done:

            # agent chooses an action, which is played
            action = agent.choose_action(available_actions)
            state_prime, reward, done, _ = env.step(action)
            # env.render()

            # env provides new actions for the agent to pick from, agent stores new knowledge and learns from it
            available_actions_prime, actions_inc_terminal_prime = env.get_after_states(include_terminal=True)
            agent.store_data(action, available_actions_prime, reward)  # (state, state_prime, reward, available_actions_prime, done)
            agent.learn()

            # update vars for the next loop
            cleared_lines += reward
            print(cleared_lines)
            state = state_prime
            available_actions, actions_inc_terminal = available_actions_prime, actions_inc_terminal_prime

        # game is over. Save stats and reset
        writer.writerow([x, cleared_lines, type(agent).__name__, run_id])
        env.reset()
        cleared_lines = 0
        done = False
        state = env.get_current_state_features()
        available_actions, actions_inc_terminal = env.get_after_states(include_terminal=True)

    # stop writing to the csv file, plot results
    open_file.close()
    df = gather_data(results_file)
    plot_gg(df, "bottom")
