#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines an environment for an agent playing Rock Paper Scissors against the
environment.
"""


import os
import sys

# TODO: is this really needed?
# Insert the package's parent directory into the system search path, so that
# this package can be imported when the aixi.py script is run directly from a
# release archive.
PROJECT_ROOT = os.path.realpath(os.path.join(os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from pyaixi import environment, util

# Define an enumeration to represent rock-paper-scissors actions, which is the
# agent performing either a rock, paper, or scissors move.
rock_paper_scissors_action_enum = util.enum("aRock", "aPaper", "aScissors")

# Define an enumeration to represent rock-paper-scissors observations, which is
# the opponent performing either a rock, paper, or scissors move.
rock_paper_scissors_observation_enum = util.enum(
    "oRock", "oPaper", "oScissors"
)

# Define an enumeration to represent losing, drawing, or winning.
rock_paper_scissors_reward_enum = util.enum("rLose", "rDraw", "rWin")

# Define some shorthand notation for ease of reference.
aRock = rock_paper_scissors_action_enum.aRock
aPaper = rock_paper_scissors_action_enum.aPaper
aScissors = rock_paper_scissors_action_enum.aScissors

oRock = rock_paper_scissors_observation_enum.oRock
oPaper = rock_paper_scissors_observation_enum.oPaper
oScissors = rock_paper_scissors_observation_enum.oScissors

rLose = rock_paper_scissors_reward_enum.rLose
rDraw = rock_paper_scissors_reward_enum.rDraw
rWin = rock_paper_scissors_reward_enum.rWin


class RockPaperScissors(environment.Environment):
    """The agent repeatedly plays Rock-Paper-Scissor against an opponent that
    has a slight, predictable bias in its strategy.

    If the opponent has won a round by playing rock on the previous cycle, it
    will always play rock at the next time step; otherwise it will pick an
    action uniformly at random.

    The agent's observation is the most recently chosen action of the opponent.

    It receives a reward of
        - `rWin` for a win,
        - `rDraw` for a draw and
        - `rLose` for a loss.

    Domain characteristics:
     - environment: "rock_paper_scissors"
     - maximum action: 2 (2 bits)
     - maximum observation: 2 (2 bits)
     - maximum reward: 2 (2 bits)
    """

    def __init__(self, options={}):
        """Construct the RockPaperScissors environment from the given options.

        - `options` is a dictionary of named options and their values."""

        # Set up the base environment.
        super().__init__(options=options)

        # Define the acceptable action values.
        self.valid_actions = list(rock_paper_scissors_action_enum.keys())

        # Define the acceptable observation values.
        self.valid_observations = list(
            rock_paper_scissors_observation_enum.keys()
        )

        # Define the acceptable reward values.
        self.valid_rewards = list(rock_paper_scissors_reward_enum.keys())

        # TODO: could the percept be represented by a class that is composed of
        #  the observation and reward?
        # Set an initial percept (i.e. not rock, to ensure a random choice in
        # the opponent on the first action.)
        self.observation = oPaper
        self.reward = 0

    def perform_action(self, action):
        """Receives the agent's action and calculates the new environment
        percept.

        (Called `performAction` in the C++ version.)"""
        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Opponent plays rock if it won the last round by playing rock,
        # otherwise it plays randomly.
        if (self.observation == aRock) and (self.reward == rLose):
            self.observation = aRock
        else:
            self.observation = util.choice(self.valid_actions)

        # TODO: logically, it doesn't make much sense to say
        #  "if action == self.observation"
        # Determine reward.
        if action == self.observation:
            # If both the agent and the opponent made the same move, it's a
            # draw.
            self.reward = rDraw
        elif action == aRock:
            # If the opponent made a scissors move, then the agent wins if
            # they played rock.
            self.reward = rWin if self.observation == oScissors else rLose
        elif action == aScissors:
            # If the opponent made a paper move, then the agent wins if they
            # played scissors.
            self.reward = rWin if self.observation == oPaper else rLose
        elif action == aPaper:
            # If the opponent made a rock move, then the agent wins if they
            # played paper.
            self.reward = rWin if self.observation == oRock else rLose
        # TODO: have an else fallback?

        # Return the resulting observation and reward.
        return self.observation, self.reward

    def print(self):
        """Returns a string indicating the status of the environment."""
        action_text = {aRock: "rock", aPaper: "paper", aScissors: "scissors"}

        observation_text = {
            oRock: "rock",
            oPaper: "paper",
            oScissors: "scissors",
        }

        reward_text = {rLose: "loses", rDraw: "draws", rWin: "wins"}

        message = (
            "Agent played "
            + action_text[self.action]
            + ", environment played "
            + observation_text[self.observation]
            + "\tAgent "
            + reward_text[self.reward]
        )

        return message
