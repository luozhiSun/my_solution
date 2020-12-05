import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np


class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
    CURRENT = 2  # indicates current cell of the agent

#action动作
class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    MOVE_LEFTUP=4
    MOVE_LEFTDOWN=5
    MOVE_RIGHTUP=6
    MOVE_RIGHTDOWM=7


class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Maze:
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.

        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at start_cell. The agent chooses actions (move left/right/up/down) in order to reach the exit_cell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.05), returning to a cell the agent visited earlier a bigger penalty (-0.25) and running into
        a wall a large penalty (-0.75). The reward (+10.0) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around clueless and looses.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlib's plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN,Action.MOVE_LEFTUP,Action.MOVE_LEFTDOWN,
               Action.MOVE_RIGHTUP,Action.MOVE_RIGHTDOWM]  # all possible actions

    reward_exit = 10.0  # 到达目标
    penalty_move = -0.05  # 没有找到退出单元格的惩罚
    penalty_visited = -0.25  # 回到之前去过的惩罚
    penalty_impossible_move = -0.75  # 到达障碍物或者是超出任务区域

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None):
        """ Create a new maze game.

            :param numpy.array maze: 2D array containing empty cells (= 0) and cells occupied with walls (= 1)
            :param tuple start_cell: starting cell for the agent in the maze (optional, else upper left)
            :param tuple exit_cell: exit cell which the agent has to reach (optional, else lower right)
        """
        self.maze = maze

        self.__minimum_reward = -2 * self.maze.size  # stop game if accumulated reward is below this threshold

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.empty.remove(self.__exit_cell)

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Variables for rendering
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves，显示运动过程
        self.__ax2 = None  # axes for rendering the best action per cell,每步中最优的动作

        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.
        将agent放在设定的初始位置中

            :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
            :return: new state after reset
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=10)  # start is a big red square
            """self.__ax1.text(*self.__current_cell, "Start", ha="center", va="center", color="white")"""
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=10)  # exit is a big green square
            """self.__ax1.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")"""
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        """ Draw a line from the agents previous cell to its current cell. """
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING): 
        """ Record what will be rendered during play and/or training.

            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.
        执行动作并收取奖惩

            :param Action action: direction in which the agent will move
            :return float: reward or penalty which results from the action
        """
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game，不能移动任何地方，迫使游戏结束
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1
            elif action==Action.MOVE_LEFTUP:
                col-=1
                row-=1
            elif action==Action.MOVE_LEFTDOWN:
                col-=1
                row+=1
            elif action==Action.MOVE_RIGHTUP:
                col+=1
                row-=1
            elif action==Action.MOVE_RIGHTDOWM:
                col+=1
                row+=1



            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__current_cell == self.__exit_cell:
                reward = Maze.reward_exit  # maximum reward when reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = Maze.penalty_visited  # penalty when returning to a cell which was visited earlier
            else:
                reward = Maze.penalty_move  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.__current_cell)
        else:
            reward = Maze.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def __possible_actions(self, cell=None):
        """ Create a list with all possible actions from 'cell', avoiding the maze's edges and walls.
        产生所有可能的动作，根据当前agent的位置

            :param tuple cell: location of the agent (optional, else use current cell)
            :return list: all possible actions
        """
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell
        """直接从所有动作中copy,通过删除不可能的动作进行初始化"""


        possible_actions = Maze.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.shape
        '''先判断执行动作是否超出限制，超出范围即remove动作，若在范围内，判断是否被占用，yes则remove'''
        '''上'''
        if row-1>=0:
            if self.maze[row-1][col]==Cell.OCCUPIED:
                possible_actions.remove(Action.MOVE_UP)
        else:
            if Action.MOVE_UP in possible_actions:
                possible_actions.remove(Action.MOVE_UP)
        if row+1<nrows:
            if self.maze[row+1][col]==Cell.OCCUPIED:
                possible_actions.remove(Action.MOVE_DOWN)
        else:
            if Action.MOVE_DOWN in possible_actions:
                possible_actions.remove(Action.MOVE_DOWN)
        if col-1>=0:
            if self.maze[row][col-1]==Cell.OCCUPIED:
                possible_actions.remove(Action.MOVE_LEFT)
        else:
            if Action.MOVE_LEFT in possible_actions:
                possible_actions.remove(Action.MOVE_LEFT)
        if col+1<ncols:
            if self.maze[row][col+1]==Cell.OCCUPIED:
                possible_actions.remove(Action.MOVE_RIGHT)
        else:
            if Action.MOVE_RIGHT in possible_actions:
                possible_actions.remove(Action.MOVE_RIGHT)
        if row-1>=0 and col-1>=0:
            if self.maze[row-1][col-1]==Cell.OCCUPIED:
                if Action.MOVE_LEFTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTUP)
        else:
            if Action.MOVE_LEFTUP in possible_actions:
                possible_actions.remove(Action.MOVE_LEFTUP)

        if row+1<nrows and col-1>=0:
            if self.maze[row+1,col-1]==Cell.OCCUPIED:
                if Action.MOVE_LEFTDOWN in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTDOWN)
        else:
            if Action.MOVE_LEFTDOWN in possible_actions:
                possible_actions.remove(Action.MOVE_LEFTDOWN)
        if row-1>=0 and col+1<ncols:
            if self.maze[row-1,col+1]==Cell.OCCUPIED:
                if Action.MOVE_RIGHTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTUP)
        else:
            if Action.MOVE_RIGHTUP in possible_actions:
                possible_actions.remove(Action.MOVE_RIGHTUP)
        if row+1<nrows and col+1<ncols:
            if self.maze[row+1,col+1]==Cell.OCCUPIED:
                if Action.MOVE_RIGHTDOWM in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTDOWM)
        else:
            if Action.MOVE_RIGHTDOWM in possible_actions:
                possible_actions.remove(Action.MOVE_RIGHTDOWM)



        """if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
            if row==0:
                if Action.MOVE_LEFTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTUP)
                if Action.MOVE_RIGHTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTUP)
                if self.maze[row-1,col]==Cell.OCCUPIED:
                    possible_actions.remove(Action.MOVE_LEFT)
                if self.maze[row+1,col]==Cell.OCCUPIED:
                    possible_actions.remove(Action.MOVE_RIGHT)
                if self.maze[row+1,col-1]==Cell.OCCUPIED:
                    possible_actions.remove(Action.MOVE_LEFTDOWN)
                if self.maze[row+1,col]==Cell.OCCUPIED:
                    possible_actions.remove()
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)
            if row==nrows-1:
                if Action.MOVE_LEFTDOWN in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTDOWN)
                if Action.MOVE_RIGHTDOWM in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTDOWM)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
            if col==0:
                if Action.MOVE_LEFTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTUP)
                if Action.MOVE_DOWN in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTDOWN)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)
            if col==ncols-1:
                if Action.MOVE_RIGHTDOWM in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTDOWM)
                if Action.MOVE_RIGHTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTUP)
        if (row>0 and row<nrows-1) and (col>0 and col<ncols-1):

            if self.maze[row - 1, col - 1] == Cell.OCCUPIED:
                if Action.MOVE_LEFTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTUP)
            if self.maze[row + 1, col - 1] == Cell.OCCUPIED:
                if Action.MOVE_LEFTDOWN in possible_actions:
                    possible_actions.remove(Action.MOVE_LEFTDOWN)
            if self.maze[row - 1, col + 1] == Cell.OCCUPIED:
                if Action.MOVE_RIGHTUP in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTUP)
            if self.maze[row + 1, col + 1] == Cell.OCCUPIED:
                if Action.MOVE_RIGHTDOWM in possible_actions:
                    possible_actions.remove(Action.MOVE_RIGHTDOWM)"""

        return possible_actions

    def __status(self):
        """ Return the game status.

            :return Status: current game status (WIN, LOSE, PLAYING)
        """
        if self.__current_cell == self.__exit_cell:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):
        """ Return the state of the maze - in this game the agents current location.

            :return numpy.array [1][2]: agents current location
        """
        return np.array([[*self.__current_cell]])

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: the prediction model to use
            :param tuple start_cell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        self.reset(start_cell)

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return status

    def check_win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = Render.NOTHING  # avoid rendering anything during execution of the check games

        win = 0
        lose = 0

        for cell in self.empty:
            if self.play(model, cell) == Status.WIN:
                win += 1
            else:
                lose += 1

        self.__render = previous  # restore previous rendering setting

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        result = True if lose == 0 else False

        return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action(s) for each cell as provided by 'model'.

        :param class AbstractModel model: the prediction model to use
        """
        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.TRAINING:
            nrows, ncols = self.maze.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.__exit_cell, "gs", markersize=10)  # exit is a big green square
            """self.__ax2.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")"""

            for cell in self.empty:
                q = model.q(cell) if model is not None else [0, 0, 0, 0,0,0,0,0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.2
                    if action == Action.MOVE_RIGHT:
                        dx = +0.2
                    if action == Action.MOVE_UP:
                        dy = -0.2
                    if action == Action.MOVE_DOWN:
                        dy = 0.2
                    if action==Action.MOVE_LEFTUP:
                        dx=-0.2
                        dy=-0.2
                    if action==Action.MOVE_LEFTDOWN:
                        dx=-0.2
                        dy=0.2
                    if action==Action.MOVE_RIGHTUP:
                        dx=0.2
                        dy=-0.2
                    if action==Action.MOVE_RIGHTDOWM:
                        dx=0.2
                        dy=0.2


                    # color (red to green) represents the certainty
                    color = clip((q[action] - -1)/(1 - -1))

                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__ax2.imshow(self.maze, cmap="binary")
            self.__ax2.get_figure().canvas.draw()
