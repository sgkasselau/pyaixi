#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for a simultaneous flip of two biased coins.
"""

from __future__ import division
from __future__ import unicode_literals

import os
import random
import sys

# Insert the package's parent directory into the system search path, so that this package can be
# imported when the aixi.py script is run directly from a release archive.
PROJECT_ROOT = os.path.realpath(os.path.join(os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from pyaixi import environment, util

# Define a enumeration to represent coin flip actions, which is a prediction of two coin landing independently on
# heads or tails.
two_coin_flip_action_enum = util.enum('aTailsTails', 'aTailsHeads', 'aHeadsTails', 'aHeadsHeads')

# Define a enumeration to represent coin flip observations e.g. the 
two_coin_flip_observation_enum = util.enum('oTailsTails', 'oTailsHeads', 'oHeadsTails', 'oHeadsHeads')

# Define a enumeration to represent coin flip rewards e.g. win or lose, for correcting predicting
# the flip coins.
two_coin_flip_reward_enum = util.enum('rLose', 'rWin')

# Define some shorthand notation for ease of reference.
aHeadsHeads = two_coin_flip_action_enum.aHeadsHeads
aHeadsTails = two_coin_flip_action_enum.aHeadsTails
aTailsHeads = two_coin_flip_action_enum.aTailsHeads
aTailsTails = two_coin_flip_action_enum.aTailsTails

oHeadsHeads = two_coin_flip_observation_enum.oHeadsHeads
oHeadsTails = two_coin_flip_observation_enum.oHeadsTails
oTailsHeads = two_coin_flip_observation_enum.oTailsHeads
oTailsTails = two_coin_flip_observation_enum.oTailsTails

observations_by_first_and_second_coin_flip = {
   0: {0: oTailsTails, 1: oTailsHeads},
   1: {0: oHeadsTails, 1: oHeadsHeads}
}

observation_text = {
    oHeadsHeads: 'heads and heads',
    oHeadsTails: 'heads and tails',
    oTailsHeads: 'tails and heads',
    oTailsTails: 'tails and tails'
}

rLose = two_coin_flip_reward_enum.rLose
rWin = two_coin_flip_reward_enum.rWin

class TwoCoinFlip(environment.Environment):
    """ Two biased coins are flipped and the agent is tasked with predicting how they
        will land. The agent receives a reward of `rWin` for a correct
        prediction and `rLoss` for an incorrect prediction. The observation
        specifies which side the both coins landed on (e.g. `oTailsTails` if both coins landed on
        tails, or `oHeadsTails` if the first coin landed on heads and the second on tails).
        The action corresponds to the agent's prediction for the
        next coin flip (`aTailsTails` or `aHeadsTails`).

        Domain characteristics:

        - environment: "two_coin_flip"
        - maximum action: 3 (2 bit)
        - maximum observation: 3 (2 bit)
        - maximum reward: 1 (1 bit)

        Configuration options:
        - `coin-flip-p`: the probability either coin lands on heads
                         Must be a number between 0 and 1 inclusive.
                         Default value is `default_probability`.
                         (Optional.)
    """

    # Instance attributes.

    # Set the default probability for the biased coin, when none is supplied from the options.
    default_probability = 0.7

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the TwoCoinFlip environment from the given options.

             - `options` is a dictionary of named options and their values.

            The following options in `options` are optional:
             - `coin-flip-p`: the probability that either coin will land on heads. (Defaults to 0.7.)
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = list(two_coin_flip_action_enum.keys())

        # Define the acceptable observation values.
        self.valid_observations = list(two_coin_flip_observation_enum.keys())

        # Define the acceptable reward values.
        self.valid_rewards = list(two_coin_flip_reward_enum.keys())

        # Determine the probability of the coin landing on heads.
        if 'coin-flip-p' not in options:
            options["coin-flip-p"] = self.default_probability
        # end if
        self.probability = float(options["coin-flip-p"])

        # Make sure the probability value is valid.
        assert 0.0 <= self.probability and self.probability <= 1.0

        # Set an initial percept by flipping two coins, then looking up the corresponding observation symbol.
        first_coin       = 1 if random.random() < self.probability else 0
        second_coin      = 1 if random.random() < self.probability else 0
        self.observation = observations_by_first_and_second_coin_flip[first_coin][second_coin]
        self.reward      = 0
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Flip the coins, and set the observation and reward appropriately.
        first_coin  = 1 if random.random() < self.probability else 0
        second_coin = 1 if random.random() < self.probability else 0
        observation = observations_by_first_and_second_coin_flip[first_coin][second_coin]
        reward      = rWin if action == observation else rLose

        # Store the observation and reward in the environment.
        self.observation = observation
        self.reward = reward

        return (observation, reward)
    # end def

    def printed(self):
        """ Returns a string indicating the status of the environment.
        """

        message = "prediction: %s" % observation_text[self.action] + \
                  ", observation: %s" % observation_text[self.observation] + \
                  ", reward: %d" % self.reward

        return message
    # end def
# end class
