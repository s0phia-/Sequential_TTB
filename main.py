import numpy as np
from tetris.game import Tetris
from agents.ttb_sequential import TakeTheBestSequential

if __name__ == "__main__":
    # This is a directed env
    env = Tetris(10, 10, False, feature_directions=[-1, -1, -1, -1, -1, -1, 1, -1])
    env.reset()
    state = env.get_current_state_features()
    available_actions = env.get_after_states()
    agent = TakeTheBestSequential(env.num_features, state)
    done = False
    cleared_lines = 0

    for x in range(100):
        env.print_current_tetromino()  ##
        action = agent.choose_action(available_actions)
        # i = np.argmax([np.sum(y) for y in after_state_features])  # equal weights
        state_prime, reward, done, _ = env.step(action)
        print(done)
        env.print_current_board()  ##
        available_actions_prime = env.get_after_states()
        agent.store_data(state, state_prime, reward, available_actions_prime, done)
        agent.learn()
        cleared_lines += reward
        print(cleared_lines)  ##
        state = state_prime
        available_actions = available_actions_prime
        if done:
            env.reset()
            # cleared_lines = 0
            state = env.get_current_state_features()
            available_actions = env.get_after_states()
