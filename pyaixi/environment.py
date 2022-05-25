#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines an environment for AIXI agents.
"""

from abc import ABC, abstractmethod

from pyaixi import util


class Environment(ABC):
    """Base class for the various agent environments.

    Each individual environment should inherit from this class and implement
    the appropriate methods.

    In particular, the constructor should set up the environment as
    appropriate, including setting the initial observation and reward, as well
    as setting appropriate values for the configuration options, via the
    `options` argument, such as

     - "agent-actions"
     - "observation-bits"
     - "reward-bits"

    Following this, the agent and environment interact in a cyclic fashion. The
    agent receives the observation and reward using `observation` and
    `reward` before supplying the environment with an action via the method
    `perform_action`, which must be implemented in subclasses.

    Upon receiving an action, the environment updates the observation and
    reward. At the beginning of each cycle, the value of `is_finished` is
    checked.

    If it is true, then there is no more interaction between the agent and the
    environment, and the program exits. Otherwise, the interaction continues
    indefinitely."""

    def __init__(self, options={}):
        """Construct an agent environment."""

        # Set the current action to None.
        # (Called `m_action` in the C++ version.)
        self.action = None

        # Set the current observation to None.
        # (Called `m_observation` in the C++ version. `getObservation` is
        # the getter method for this property in the C++ version.)
        self.observation = None

        # Set the current reward to None.
        # (Called `m_reward` in the C++ version. `getReward` is the getter
        # method for this property in the C++ version.)
        self.reward = None

        # The C++ version does not define the next 3 lists, but, instead, it
        # defines the virtual methods `isValidAction`,
        # `isValidObservation` and `isValidReward`, which return true if
        # the given input (action, observation and reward, respectively) is
        # between the minimum and maximum values for the action, observation
        # and reward, respectively; else false.

        # Defines the acceptable action values.
        self.valid_actions = []

        # Define the acceptable observation values.
        self.valid_observations = []

        # Define the acceptable reward values.
        self.valid_rewards = []

        # Set whether the environment is finished.
        # (This field corresponds to the method `isFinished` in the C++
        # version.)
        self.is_finished = False

        # Store the given options.
        self.options = options

    @abstractmethod
    def perform_action(self, action):
        """Receives the agent's action and calculates the new environment
        percept.

        (Called `performAction` in the C++ version.)"""
        pass

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

        (Called `maxAction` in the C++ version. In the C++ version, it's
        a pure virtual (aka abstract) function that must be overridden by
        subclasses.)"""

        # The largest action is the last in the list of valid actions.
        # Else, it's None.
        # TODO: see function minimum_action.
        return self.valid_actions[-1] if len(self.valid_actions) > 0 else None

    def maximum_observation(self):
        """Returns the maximum possible observation.

        (Called `maxObservation` in the C++ version. In the C++ version, it's
        a pure virtual (aka abstract) function that must be overridden by
        subclasses.)"""

        # The largest observation is the last in the list of valid
        # observations. Else, it's None.
        # TODO: see function minimum_action.
        return (
            self.valid_observations[-1]
            if len(self.valid_observations) > 0
            else None
        )

    def maximum_reward(self):
        """Returns the maximum possible reward.

        (Called `maxReward` in the C++ version. In the C++ version, it's
        a pure virtual (aka abstract) function that must be overridden by
        subclasses.)"""

        # The largest reward is the last in the list of valid rewards.
        # Else, it's None.
        # TODO: see function minimum_action.
        return self.valid_rewards[-1] if len(self.valid_rewards) > 0 else None

    def minimum_action(self):
        """Returns the minimum possible action.

        (Called `minAction` in the C++ version.)"""

        # The smallest action is the first in the list of valid actions.
        # Else, it's None.
        # TODO: this can lead to an index error: len(self.valid_actions) could
        #  have just one element, so self.valid_actions[1] would be an index
        #  error. There are other examples. Maybe they mean -1 instead of 1?
        #  See function maximum_reward, where -1 is used.
        return self.valid_actions[1] if len(self.valid_actions) > 0 else None

    def minimum_observation(self):
        """Returns the minimum possible observation.

        (Called `minObservation` in the C++ version.)"""

        # The smallest observation is the first in the list of valid
        # observations. Else, it's None.
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
        # Else, it's None.
        # TODO: could be an index error. See function minimum_action
        return self.valid_rewards[1] if len(self.valid_rewards) > 0 else None

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

    def percept_bits(self):
        """Returns the maximum number of bits required to represent a percept.

        (Called `perceptBits` in the C++ version.)"""
        return self.observation_bits() + self.reward_bits()

    def __str__(self):
        """Returns a string representation of this environment instance."""
        return (
            "action = "
            + str(self.action)
            + ", observation = "
            + str(self.observation)
            + ", reward = "
            + str(self.reward)
        )

    def print(self):
        """String representation convenience method from the C++ version."""
        return str(self)
