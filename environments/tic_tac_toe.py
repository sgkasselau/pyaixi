#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for Tic Tac Toe.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random

from pyaixi import environment, util

# Define a enumeration to represent environment observations: either a square
# is empty, filled with the agent's piece, or the environment's piece.
tictactoe_observation_enum = util.enum('oEmpty', 'oAgent', 'oEnv')

# Define a enumeration to represent rewards as a result of actions: invalid actions,
# losses, null, draws, and wins.
# NOTE: since the enumeration values need to be positive, these values are defined relative to 3.
#       So -3 points is 0, -2 points is 1, and 2 points is 5.
tictactoe_reward_enum = util.enum(rInvalid = 0, rLoss = 1, rNull = 3, rDraw = 4, rWin = 5)

# Define some shorthand notation for ease of reference.
oEmpty  = tictactoe_observation_enum.oEmpty
oAgent  = tictactoe_observation_enum.oAgent
oEnv    = tictactoe_observation_enum.oEnv

rInvalid = tictactoe_reward_enum.rInvalid
rLoss    = tictactoe_reward_enum.rLoss
rNull    = tictactoe_reward_enum.rNull
rDraw    = tictactoe_reward_enum.rDraw
rWin     = tictactoe_reward_enum.rWin

class Tic_Tac_Toe(environment.Environment):
    """ In this domain, the agent plays repeated games of TicTacToe against an
        opponent who moves randomly. If the agent wins the game, it receives a
        reward of 2. If there is a draw, the agent receives a reward of 1. A loss
        penalizes the agent by -2. If the agent makes an illegal move, by moving on
        top of an already filled square, then it receives a reward of -3. A legal
        move that does not end the game earns no reward. An illegal reward causes
        the game to restart.

        Domain characteristics:
        - environment: "tictactoe"
        - maximum action: 8 (4 bits)
        - maximum observation: 174672 (18 bits)
          - 174672 (decimal) = 101010101010101010 (binary)
        - maximum reward: 5 (3 bits)
    """

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the Tic Tac Toe environment from the given options.

             - `options` is a dictionary of named options and their values.
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = xrange(0, 9)

        # Define the acceptable observation values.
        self.valid_observations = xrange(0, 174672 + 1)

        # Define the acceptable reward values.
        self.valid_rewards = tictactoe_reward_enum.keys()

        # Set the initial reward.
        self.reward = 0

        # Set up the game.
        self.reset()
    # end def

    def check_win(self):
        """ Check if either player has won the game.
            Returns True if so, False otherwise.

            (Called `checkWin` in the C++ version.)
        """

        # Check if we've got a row of three matching symbols.
        for r in xrange(0, 3):
            # Is this row all matching, non-empty symbols?
            if (self.board[r][0] != oEmpty and \
                self.board[r][0] == self.board[r][1] and \
                self.board[r][1] == self.board[r][2]):
                # Yes. Someone has won.
                return True
            # end if
        # end for

        # Check if we've got any columns of three matching symbols.
        for c in xrange(0, 3):
            # Is this column all matching, non-empty symbols?
            if (self.board[0][c] != oEmpty and \
                self.board[0][c] == self.board[1][c] and \
                self.board[1][c] == self.board[2][c]):
                # Yes. Someone has won.
                return True
            # end if
        # end for

        # Check the diagonals.
        if (self.board[1][1] != oEmpty and \
            self.board[0][0] == self.board[1][1] and \
            self.board[1][1] == self.board[2][2]):
            return True
        # end if

        if (self.board[1][1] != oEmpty and \
            self.board[0][2] == self.board[1][1] and \
            self.board[1][1] == self.board[2][0]):
            return True
        # end if

        # If we're here, there's no winner yet.
        return False
    # end def

    def compute_observation(self):
        """ Encodes the state of each square into an overall observation and saves
            the result in self.observation.
            Each cell corresponds to two bits.

            (Called `computeObservation` in the C++ version.)
        """

        self.observation = 0
        for r in xrange(0, 3):
            for c in xrange(0, 3):
                # Shift the existing observation up by 2 bits, and add the current observation to
                # that.
                self.observation = self.board[r][c] + (4 * self.observation)
            # end for
        # end for
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Increment the actions-since-reset counter.
        self.actions_since_reset += 1

        # Decode the action into the desired row and column to update.
        r = int(action / 3)
        c = action % 3

        # If agent makes an invalid move, give the appropriate (lack of) reward and clear the board.
        if (self.board[r][c] != oEmpty):
            self.reward = rInvalid
            self.reset()
            return
        # end def

        # The agent makes their move.
        self.board[r][c] = oAgent

        # If the agent wins or draws, give an appropriate reward and clear the board.
        if (self.check_win()):
            self.reward = rWin
            self.reset()
            return
        elif self.actions_since_reset == 5:
            # If it's been 5 actions since the reset, then this must be a draw, as there's
            # no way to win from this point.
            self.reward = rDraw
            self.reset()
            return
        # end def

        # The environment makes a random play.
        while (self.board[r][c] != oEmpty):
            # Keep picking board positions at random until we find an unoccupied spot.
            r = random.randrange(0, 3)
            c = random.randrange(0, 3)
        # end while

        # If we're here, we've got an unoccupied spot.
        self.board[r][c] = oEnv

        # If the environment has won, give an appropriate reward and clear the board.
        if (self.check_win()):
            self.reward = rLoss
            self.reset()
            return
        # end def

        # The game continues.
        self.reward = rNull
        self.compute_observation()

        return (self.observation, self.reward)
    # end def

    def print(self):
        """ Returns a string indicating the status of the environment.
        """

        # Show what just happened, correcting the reward for being defined relative to 3.
        message = "action = %s, observation = %s, reward = %s (%d), board:" % \
                  (self.action, self.observation, self.reward, (self.reward - 3)) + os.linesep

        # Display the current state of the board.
        for r in xrange(0, 3):
            for c in xrange(0, 3):
                b = self.board[r][c]
                message += "." if b == oEmpty else ("A" if b == oAgent else "O")
            # end for
            message += os.linesep
        # end for
        message += os.linesep

        return message
    # end def

    def reset(self):
        """ Begin a new game.
        """

        # Set up the board.
        self.board = {}

        for r in xrange(0, 3):
            for c in xrange(0, 3):
                # Ensure the row exists.
                if r not in self.board:
                    self.board[r] = {}
                # end if

                # Set this element to be empty.
                self.board[r][c] = oEmpty
            # end for
        # end for

        # Set an initial observation.
        self.compute_observation()

        # Set the actions-since-reset marker.
        self.actions_since_reset = 0
    # end def
# end class