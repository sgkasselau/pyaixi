#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines an environment for AIXI agents.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

from pyaixi import util


class Environment:
    """Base class for the various agent environments.

    Each individual environment should inherit from this class and implement
    the appropriate methods.

    In particular, the constructor should set up the environment as
    appropriate, including setting the initial observation and reward, as
    well as setting appropriate values for the configuration options:

    - `agent-actions`
    - `observation-bits`
    - `reward-bits`

    Following this, the agent and environment interact in a cyclic fashion. The
    agent receives the observation and reward using
    `Environment.getObservation` and `Environment.getReward` before supplying
    the environment with an action via `Environment.performAction`.

    Upon receiving an action, the environment updates the observation and
    reward. At the beginning of each cycle, the value of
    `Environment::isFinished` is checked.

    If it is true then there is no more interaction between the agent and
    environment, and the program exits. Otherwise, the interaction continues
    indefinitely."""

    def __init__(self, options={}):
        """Construct an agent environment."""

        # Set the current action to null/None.
        # (Called `m_action` and `getAction` in the C++ version.)
        self.action = None

        # Set whether the environment is finished.
        # (Called `isFinished` in the C++ version.)
        self.is_finished = False

        # Set the current observation to null/None.
        # (Called `m_observation` and `getObservation()` in the C++ version.)
        self.observation = None

        # Store the given options.
        self.options = options

        # Set the current reward to null/None.
        # (Called `m_reward` in the C++ version.)
        self.reward = None

        # Defines the acceptable action values.
        self.valid_actions = []

        # Define the acceptable observation values.
        self.valid_observations = []

        # Define the acceptable reward values.
        self.valid_rewards = []

    def __unicode__(self):
        """Returns a string representation of this environment instance."""
        return (
            "action = "
            + str(self.action)
            + ", observation = "
            + str(self.observation)
            + ", reward = "
            + str(self.reward)
        )

    # TODO: get rid of this!
    # Make a compatible string function for the current Python version.
    if sys.version_info[0] >= 3:
        # For Python 3.
        def __str__(self):
            return self.__unicode__()

    else:
        # For Python 2.
        def __str__(self):
            return self.__unicode__().encode("utf8")

    def action_bits(self):
        """Returns the maximum number of bits required to represent an action.

        (Called `actionBits` in the C++ version.)"""

        # Find the largest sized observation.
        maximum_bits = 0
        for action in self.valid_actions:
            bits_for_this_action = util.bits_required(action)
            if bits_for_this_action > maximum_bits:
                maximum_bits = bits_for_this_action

        return maximum_bits

    def is_valid_action(self, action):
        """Returns whether the given action is valid.

        (Called `isValidAction` in the C++ version.)"""
        return action in self.valid_actions

    def is_valid_observation(self, observation):
        """Returns whether the given observation is valid.

        (Called `isValidObservation` in the C++ version.)"""
        return observation in self.valid_observations

    def is_valid_reward(self, reward):
        """Returns whether the given reward is valid.

        (Called `isValidReward` in the C++ version.)"""
        return reward in self.valid_rewards

    def maximum_action(self):
        """Returns the maximum possible action.

        (Called `maxAction` in the C++ version.)"""

        # The largest action is the last in the list of valid actions.
        # Else, it's null/None.
        # TODO: see function minimum_action.
        return self.valid_actions[-1] if len(self.valid_actions) > 0 else None

    def maximum_observation(self):
        """Returns the maximum possible observation.

        (Called `maxObservation` in the C++ version.)"""

        # The largest observation is the last in the list of valid
        # observations. Else, it's null/None.
        # TODO: see function minimum_action.
        return (
            self.valid_observations[-1]
            if len(self.valid_observations) > 0
            else None
        )

    def maximum_reward(self):
        """Returns the maximum possible reward.

        (Called `maxReward` in the C++ version.)"""

        # The largest reward is the last in the list of valid rewards.
        # Else, it's null/None.
        # TODO: see function minimum_action.
        return self.valid_rewards[-1] if len(self.valid_rewards) > 0 else None

    def minimum_action(self):
        """Returns the minimum possible action.

        (Called `minAction` in the C++ version.)"""

        # The smallest action is the first in the list of valid actions.
        # Else, it's null/None.
        # TODO: this can lead to an index error: len(self.valid_actions) could
        #  have just one element, so self.valid_actions[1] would be an index
        #  error. There are other examples. Maybe they mean -1 instead of 1?
        #  See functoin maximum_reward, where -1 is used.
        return self.valid_actions[1] if len(self.valid_actions) > 0 else None

    def minimum_observation(self):
        """Returns the minimum possible observation.

        (Called `minObservation` in the C++ version.)"""

        # The smallest observation is the first in the list of valid
        # observations. Else, it's null/None.
        # TODO: could be an index error. See function minimum_action
        return (
            self.valid_observations[1]
            if len(self.valid_observations) > 0
            else None
        )

    def minimum_reward(self):
        """Returns the minimum possible reward.

        (Called `minReward` in the C++ version.)"""

        # The smallest reward is the first in the list of valid rewards.
        # Else, it's null/None.
        # TODO: could be an index error. See function minimum_action
        return self.valid_rewards[1] if len(self.valid_rewards) > 0 else None

    def observation_bits(self):
        """Returns the maximum number of bits required to represent an
        observation.

        (Called `observationBits` in the C++ version)"""

        # Find the largest sized observation.
        maximum_bits = 0
        for observation in self.valid_observations:
            bits_for_this_observation = util.bits_required(observation)
            if bits_for_this_observation > maximum_bits:
                maximum_bits = bits_for_this_observation

        return maximum_bits

    def percept_bits(self):
        """Returns the maximum number of bits required to represent a percept.

        (Called `perceptBits` in the C++ version.)"""
        return self.observation_bits() + self.reward_bits()

    def perform_action(self, action):
        """Receives the agent's action and calculates the new environment
        percept.

        (Called `performAction` in the C++ version.)"""
        # To be overridden by inheriting classes.
        pass

    def print(self):
        """String representation convenience method from the C++ version."""
        return str(self)

    def reward_bits(self):
        """Returns the maximum number of bits required to represent a reward.

        (Called `rewardBits` in the C++ version)"""

        # Find the largest sized reward.
        maximum_bits = 0
        for reward in self.valid_rewards:
            bits_for_this_reward = util.bits_required(reward)
            if bits_for_this_reward > maximum_bits:
                maximum_bits = bits_for_this_reward

        return maximum_bits
