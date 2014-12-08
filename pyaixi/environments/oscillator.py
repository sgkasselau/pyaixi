#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for an agent interacting with an environment which oscillates between two values.
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

# Define a enumeration to represent agent interactions with the environment,
# such as guessing high or low.
oscillator_action_enum = util.enum(aLow = 0, aHigh = 1)

# Define a enumeration to represent environment observations: either being
# in the low state, or the high state.
oscillator_observation_enum = util.enum(oLow = 0, oHigh = 1)

# Define a enumeration to represent rewards as a result of actions: guessing right,
# or guessing wrong.
oscillator_reward_enum = util.enum(rWrong = 0, rRight = 1)

# Define some shorthand notation for ease of reference.
aLow   = oscillator_action_enum.aLow
aHigh  = oscillator_action_enum.aHigh

oLow   = oscillator_observation_enum.oLow
oHigh  = oscillator_observation_enum.oHigh

rWrong  = oscillator_reward_enum.rWrong
rRight  = oscillator_reward_enum.rRight

class Oscillator(environment.Environment):
    """ A two-value (low and high) oscillatorer.

        The observations are the current value, and actions performed by the agent
        should match the current value, in order to earn a small reward.

        Domain characteristics:

        - environment: "oscillator"
        - maximum action: 1 (1 bit)
        - maximum observation: 1 (1 bit)
        - maximum reward: 1 (1 bit)

        Configuration options:
        - `oscillator-delay`: the number of rounds to wait before changing the oscillation state.

                             Must be an integer.

                             Default value is `default_delay`, 1.
                             (Optional.)
    """

    # Instance attributes.

    # Set the default oscillation delay.
    default_delay = 1

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the Oscillator environment from the given options.

             - `options` is a dictionary of named options and their values.

            The following options in `options` are optional:
             - `oscillator-delay`: the number of rounds to wait before changing the oscillation state.
                                   Default value: 1.
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = list(oscillator_action_enum.keys())

        # Define the acceptable observation values.
        self.valid_observations = list(oscillator_observation_enum.keys())

        # Define the acceptable reward values.
        self.valid_rewards = list(oscillator_reward_enum.keys())

        # Set the delay value.
        if 'oscillator-delay' not in options:
            options["oscillator-delay"] = self.default_delay
        # end if
        self.delay = int(options["oscillator-delay"])

        # Set a count of rounds since the last oscillation.
        self.rounds = 0

        # Set an initial percept.
        self.observation = oLow
        self.reward = 0
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Increment the number of rounds since the last oscillation.
        self.rounds += 1

        # Is it time to oscillate the state?
        if self.rounds >= self.delay:
            self.observation = oLow if (self.observation == oHigh) else oHigh
            self.rounds = 0
        # end if

        # If the action matches the oscillation state, reward the agent.
        if (action == self.observation):
            self.reward = 1
        else:
            self.reward = 0
        # end if

        return (self.observation, self.reward)
    # end def

    def printed(self):
        """ Returns a string indicating the status of the environment.
        """

        action_text      = {aLow:  "low",
                            aHigh: "high"}
        observation_text = {oLow:  "low",
                            oHigh: "high"}
        reward_text      = {rRight: "right!",
                            rWrong: "wrong"}

        # Show what just happened, correcting the reward for being defined relative to 100.
        message = "action = %s, observation = %s, reward = %s (%d)" % \
                  (action_text[self.action],
                   observation_text[self.observation],
                   reward_text[self.reward],
                   self.reward)

        return message
    # end def
# end class