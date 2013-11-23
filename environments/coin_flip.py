#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for a biased coin flip.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from pyaixi import environment, util

# Define a enumeration to represent coin flip actions, which is a prediction of the coin landing on
# heads or tails.
coin_flip_action_enum = util.enum('aTails', 'aHeads')

# Define a enumeration to represent coin flip observations e.g. the coin landed on heads or tails.
coin_flip_observation_enum = util.enum('oTails', 'oHeads')

# Define a enumeration to represent coin flip rewards e.g. win or lose, for correcting predicting
# the coin flip.
coin_flip_reward_enum = util.enum('rLose', 'rWin')

# Define some shorthand notation for ease of reference.
aHeads = coin_flip_action_enum.aHeads
aTails = coin_flip_action_enum.aTails

oHeads = coin_flip_observation_enum.oHeads
oTails = coin_flip_observation_enum.oTails

rLose = coin_flip_reward_enum.rLose
rWin = coin_flip_reward_enum.rWin

class CoinFlip(environment.Environment):
    """ A biased coin is flipped and the agent is tasked with predicting how it
        will land. The agent receives a reward of `rWin` for a correct
        prediction and `rLoss` for an incorrect prediction. The observation
        specifies which side the coin landed on (`oTails` or `oHeads`).
        The action corresponds to the agent's prediction for the
        next coin flip (`aTails` or `aHeads`).

        Domain characteristics:

        - environment: "coin_flip"
        - maximum action: 1 (1 bit)
        - maximum observation: 1 (1 bit)
        - maximum reward: 1 (1 bit)

        Configuration options:
        - `coin-flip-p`: the probability the coin lands on heads
                         (`oHeads`). Must be a number between 0 and 1 inclusive.
                         Default value is `default_probability`.
                         (Optional.)
    """

    # Instance attributes.

    # Set the default probability for the biased coin, when none is supplied from the options.
    default_probability = 0.7

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the CoinFlip environment from the given options.

             - `options` is a dictionary of named options and their values.

            The following options in `options` are optional:
             - `coin-flip-p`: the probability that the coin will land on heads. (Defaults to 0.7.)
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = coin_flip_action_enum.keys()

        # Define the acceptable observation values.
        self.valid_observations = coin_flip_observation_enum.keys()

        # Define the acceptable reward values.
        self.valid_rewards = coin_flip_reward_enum.keys()

        # Determine the probability of the coin landing on heads.
        if 'coin-flip-p' not in options:
            options["coin-flip-p"] = self.default_probability
        # end if
        self.probability = float(options["coin-flip-p"])

        # Make sure the probability value is valid.
        assert 0.0 <= self.probability and self.probability <= 1.0

        # Set an initial percept.
        self.observation = oHeads if random.random() < self.probability else oTails
        self.reward = 0
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Flip the coin, set observation and reward appropriately.
        if (random.random() < self.probability):
            observation = oHeads
            reward = rWin if action == oHeads else rLose
        else:
            observation = oTails
            reward = rWin if action == oTails else rLose
        # end if

        # Store the observation and reward in the environment.
        self.observation = observation
        self.reward = reward

        return (observation, reward)
    # end def

    def print(self):
        """ Returns a string indicating the status of the environment.
        """

        message = "prediction: " + \
                  ("tails" if self.action == aTails else "heads") + \
                  ", observation: " + \
                  ("tails" if self.observation == oTails else "heads") + \
                  ", reward: %d" % self.reward

        return message
    # end def
# end class
