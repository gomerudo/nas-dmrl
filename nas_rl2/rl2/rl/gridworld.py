"""Definition of the Grid World Environment.

This is the environment file, is taken from https://github.com/dennybritz/
reinforcement-learning/blob/master/lib/envs/gridworld.py
"""

import sys
import numpy as np
from gym.envs.toy_text import discrete

# We define the constants for the actions. Only 4 actions in this environment.
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """The Grid World environment.

    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """
    # Metadata is only the render modes. we can just ignore it for now...
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4, 4]):
        # Check that shape is valid: 2D
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        # Store the shape, of the environment
        self.shape = shape

        # total number of possibles states in the environment: 4x4
        nS = np.prod(shape)
        # only four actions: UP, DOWN, RIGHT, LEFT
        nA = 4

        # Define boundaries for the logical array representing the environment
        MAX_Y = shape[0]
        MAX_X = shape[1]

        # P will store the array of (Tr_prob, next_state, reward, Term_flag)
        # for every state s (16 states in the default)
        P = {}
        # Create a grid from 0...15 (default) and reshape it to the 
        # environment's shape
        grid = np.arange(nS).reshape(shape)
        # multi index will create "indices" with the position of the element
        # in the original ndarray (0,0), (0,1), (1,0), etc...
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex  # the actual state: 0,1,2,3...
            y, x = it.multi_index # The coordinate in the original ndarray

            # P will contain, per every state (0, 1, 2...) an array of size
            # n_actions, i.e. every state has all actions.
            P[s] = {a: [] for a in range(nA)}

            # Is done is a function that will evaluate whether or not a given
            # state is the first or the last one.
            is_done = lambda s: s == 0 or s == (nS - 1)
            # reward is 0 if is the last state, -1 if is not.
            reward = 0.0 if is_done(s) else -1.0

            # If we are at the beginning or at the end of the states (0 or 15)
            # That actually means: start or exit in the environment
            if is_done(s):
                # The state-action will be equal for everyone:
                # Tr_prob=1.0, next_state=current_state, reward=0.0,
                # termination_flag=True
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                # Calculate next state for every possible action
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                # Assigng the corresponding next state to every action.
                # Tr_prob=1.0, next_state=ns_*, reward=-1.0,
                # term_flag=whether or not next_state is a terminal state.
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            # Move to the next element: it.next()
            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)


    def _render(self, mode='human', close=False):
        """Show the environment's current snapshot."""
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
