import numpy as np
import random


class ChasingGridWorld:
    """
    Gridworld with a monster that chases the agent with some probability, behaves randomly the rest of the time
    """

    def __init__(self, cols, rows):
        self.num_cols = cols
        self.num_rows = rows
        self.agent_position, self.monster_position = self.reset()
        self.state = self.get_state()
        self.random_monster = .1

    def reset(self):
        """
        set the agent position and monster position to random (distinct) locations on the grid
        :return: agent and monster's new positions
        """

        def random_grid_position(class_self):  # get a random grid position
            return np.array((random.randint(0, class_self.num_rows - 1), random.randint(0, class_self.num_cols - 1)))

        self.agent_position = random_grid_position(self)
        self.monster_position = random_grid_position(self)

        while (self.monster_position == self.agent_position).all():  # ensure monster and agent positions are not same
            self.monster_position = random_grid_position(self)

        return self.agent_position, self.monster_position

    def step(self, action: list):
        """
        Move the agent according to the action, then move the monster to chase the agent. Game finishes if the agent
        and mosnter are ever on the same square
        :param action: the [x, y] directions the agent should move in
        :return: the state with the agent's and monster's new position
        """

        def check_action_valid(which_agent, act, class_self=self):
            if which_agent == "agent":
                position = class_self.agent_position
            elif which_agent == "monster":
                position = class_self.monster_position
            else:
                return "invalid which agent"
            # check whether the action moves the agent/monster outside the bounds of the grid
            if (position + np.array(act, dtype=int) >= [class_self.num_rows, class_self.num_cols]).any() \
                    or (position + np.array(act, dtype=int) < [0, 0]).any():
                # agent is trying to move outside the bounds of the grid
                return False
            else:
                return True

        # only move the agent if the action is valid. Otherwise, action does nothing
        if check_action_valid("agent", action) is True:
            self.agent_position += action

        if (self.agent_position == self.monster_position).all():  # if the move takes the agent into the monster
            done = True  # game is over
            reward = -100
            state = self.get_state()
            return state, reward, done, ""

        monster_action = self.get_monster_action()

        # only move monster if action is valid. Otherwise action does nothing
        if check_action_valid("monster", monster_action) is True:
            self.monster_position += np.array(monster_action, dtype=int)
        state = self.get_state()

        # if the agent and monster share a location, the game is over
        if (self.agent_position == self.monster_position).all():
            done = True
            reward = -100
        else:
            done = False
            reward = 1

        return state, reward, done, ""

    def get_monster_action(self):
        if random.uniform(0, 1) < self.random_monster:  # if the monster behaves randomly
            monster_action_x = random.randint(0, 1)
            monster_action_y = 1 - monster_action_x
            monster_action = (monster_action_x, monster_action_y)

        else:  # monster chases the agent - it moves 1 square towards the agent on the axis with the biggest difference
            diff = self.agent_position - self.monster_position
            biggest_diff_axis = np.argmax([abs(diff)])
            monster_action = [0, 0]
            monster_action[biggest_diff_axis] = diff[biggest_diff_axis] / abs(diff[biggest_diff_axis])  # gets +/- 1

        return monster_action

    def render(self):
        """
        Print a grid with 0s everywhere except 1 in the agent's position and 2 in the monster's position
        """
        grid = np.zeros([self.num_rows, self.num_cols])
        grid[self.agent_position[0], self.agent_position[1]] = 1
        grid[self.monster_position[0], self.monster_position[1]] = 2
        print(grid)

    def get_state(self):
        """
        Calculate the current state. The four features are
        agent_x - monster_x
        agent_y - monster_y
        monster_x - agent_x
        monster_y - agent_y
        :return: current state
        """
        state = np.zeros(4)
        state[0:2] = self.agent_position - self.monster_position
        state[2:4] = self.monster_position - self.agent_position
        self.state = state
        return state
