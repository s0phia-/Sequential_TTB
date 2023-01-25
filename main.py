from tetris.game import Tetris
from agents.ttb_q import TTB_Q_values
from agents.pure_ew import EqualWeights
from agents.hand_crafted import HandCrafted
from agents.ttb_validities import TTB_validities
from agents.ttb_rollout_validities import TTB_Roll_Val, TTB_Roll_Cond_Val
from agents.ttb_rollouts import TTB_Rollouts
from analysis.plot import gather_data, plot_gg, results_writer
from run_files import run_ttb_val, run_ttb_q, run_ttb_rollouts, run_simple


if __name__ == "__main__":

    # write results to CSV files
    results_file = 'results/result1.csv'
    open_file, writer, run_id = results_writer(results_file)

    # Create a tetris env with directed features
    env = Tetris(10, 10, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()

    # create an agent (may need some info from the environment)
    state = env.get_current_state_features()
    agent = TTB_Roll_Cond_Val(env.num_features, state)

    # set up play loop
    num_episodes = 15

    run_ttb_rollouts(agent, env, num_episodes, writer, run_id)

    # stop writing to the csv file, plot results
    open_file.close()
    df = gather_data(results_file)
    plot_gg(df, "bottom")
