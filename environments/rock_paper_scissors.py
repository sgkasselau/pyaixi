#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for an agent playing Rock Paper Scissors against the environment.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from pyaixi import environment, util

# Define a enumeration to represent rock-paper-scissors actions, which is the
# agent performing either a rock, paper, or a scissors move.
rock_paper_scissors_action_enum = util.enum('aRock', 'aPaper', 'aScissors')

# Define a enumeration to represent rock-paper-scissors observations, which is the
# opponent performing either a rock, paper, or a scissors move.
rock_paper_scissors_observation_enum = util.enum('oRock', 'oPaper', 'oScissors')

# Define a enumeration to represent losing, drawing, or winning.
rock_paper_scissors_reward_enum = util.enum('rLose', 'rDraw', 'rWin')

# Define some shorthand notation for ease of reference.
aRock     = rock_paper_scissors_action_enum.aRock
aPaper    = rock_paper_scissors_action_enum.aPaper
aScissors = rock_paper_scissors_action_enum.aScissors

oRock     = rock_paper_scissors_observation_enum.oRock
oPaper    = rock_paper_scissors_observation_enum.oPaper
oScissors = rock_paper_scissors_observation_enum.oScissors

rLose     = rock_paper_scissors_reward_enum.rLose
rDraw     = rock_paper_scissors_reward_enum.rDraw
rWin      = rock_paper_scissors_reward_enum.rWin


class RockPaperScissors(environment.Environment):
    """ The agent repeatedly plays Rock-Paper-Scissor against an opponent that has
        a slight, predictable bias in its strategy.

        If the opponent has won a round by playing rock on the previous cycle, it
        will always play rock at the next time step; otherwise it will pick an
        action uniformly at random.

        The agent's observation is the most recently chosen action of the opponent.
        It receives a reward of `rWin` for a win, `rDraw` for a draw and `rLose` for a loss.

        Domain characteristics:
         - environment: "rock_paper_scissors"
         - maximum action: 2 (2 bits)
         - maximum observation: 2 (2 bits)
         - maximum reward: 2 (2 bits)
    """

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the RockPaperScissors environment from the given options.

             - `options` is a dictionary of named options and their values.
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = rock_paper_scissors_action_enum.keys()

        # Define the acceptable observation values.
        self.valid_observations = rock_paper_scissors_observation_enum.keys()

        # Define the acceptable reward values.
        self.valid_rewards = rock_paper_scissors_reward_enum.keys()

        # Set an initial percept.
        # (i.e. not rock, to ensure a random choice in the opponent on the first action.)
        self.observation = oPaper
        self.reward = 0
    # end def


    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.  
        self.action = action

        # Opponent plays rock if it won the last round by playing rock, otherwise
        # it plays randomly.
        if self.observation == aRock and self.reward == rLose:
            self.observation = aRock
        else:
            self.observation = random.choice(self.valid_actions)
        # end if

        # Determine reward.
        if action == self.observation:
            # If both the agent and the opponent made the same move, it's a draw.
            self.reward = rDraw
        elif action == aRock:
            # If the opponent made a scissors move, then the agent wins if they played rock.
            self.reward = rWin if self.observation == oScissors else rLose
        elif action == aScissors:
            # If the opponent made a paper move, then the agent wins if they played scissors.
            self.reward = rWin if self.observation == oPaper else rLose
        elif action == aPaper:
            # If the opponent made a rock move, then the agent wins if they played paper.
            self.reward = rWin if self.observation == oRock else rLose
        # end if

        # Return the resulting observation and reward.
        return (self.observation, self.reward)
    # end def

    def print(self):
        """ Returns a string indicating the status of the environment.
        """

        action_text      = {aRock: "rock", aPaper: "paper", aScissors: "scissors"}
        observation_text = {oRock: "rock", oPaper: "paper", oScissors: "scissors"}
        reward_text      = {rLose: "loses", rDraw: "draws", rWin: "wins"}

        message = "Agent played " + action_text[self.action] + ", " + \
                  "environment played " + observation_text[self.observation] + "\t" + \
                  "Agent " + reward_text[self.reward]

        return message
    # end def
# end class