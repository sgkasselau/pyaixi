#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for an agent interacting with an environment where there's
a tiger and a pot of gold hidden separately, behind two closed doors.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from pyaixi import environment, util

# Define a enumeration to represent agent interactions with the environment,
# such as listening, or opening doors.
tiger_action_enum = util.enum('aListen', 'aLeft', 'aRight')

# Define a enumeration to represent environment observations: either not
# hearing the tiger, or hearing it from behind either door.
tiger_observation_enum = util.enum('oNull', 'oLeft', 'oRight')

# Define a enumeration to represent rewards as a result of actions: being eaten by the tiger,
# getting information from listening, or finding the gold
# NOTE: since the enumeration values need to be positive, these values are defined relative to
#       100.
#       So -100 points is 0, -1 points is 99, and 10 points is 110.
tiger_reward_enum = util.enum(rEaten = 0, rListen = 99, rGold = 110)

# Define some shorthand notation for ease of reference.
aListen = tiger_action_enum.aListen
aLeft   = tiger_action_enum.aLeft
aRight  = tiger_action_enum.aRight

oNull  = tiger_observation_enum.oNull
oLeft  = tiger_observation_enum.oLeft
oRight = tiger_observation_enum.oRight

rEaten  = tiger_reward_enum.rEaten
rListen = tiger_reward_enum.rListen
rGold   = tiger_reward_enum.rGold

class Tiger(environment.Environment):
    """ The environment dynamics are as follows: a tiger and a pot of gold are
        hidden behind one of two doors.

        Initially the agent starts facing both doors. The agent has a choice of
        one of three actions: listen, open the left door, or open the right door.

        If the agent opens the door hiding the tiger, it suffers a -100 penalty.
        If it opens the door with the pot of gold, it receives a reward of 10.

        If the agent performs the listen action, it receives a penalty of -1 and
        an observation that correctly describes where the tiger is with 0.85 probability.

        Domain characteristics:

        - environment: "tiger"
        - maximum action: 2 (2 bits)
        - maximum observation: 2 (2 bits)
        - maximum reward: 110 (7 bits)

        Configuration options:
        - `tiger-listen-accuracy`: the probability that listening for the
                                   tiger results in the correct observation of the tiger's
                                   whereabouts (`listen_accuracy`).

                                   Must be a floating point number between 0.0
                                   and 1.0 inclusive.

                                   Default value is `default_listen_accuracy`, 85%.
                                   (Optional.)
    """

    # Instance attributes.

    # Set the default probability for the listen accuracy.
    default_listen_accuracy = 0.85

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the Tiger environment from the given options.

             - `options` is a dictionary of named options and their values.

            The following options in `options` are optional:
             - `tiger-listen-accuracy`: the probability of correctly observing the tiger's location from listening.
                                        Default value: 85%.
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = tiger_action_enum.keys()

        # Define the acceptable observation values.
        self.valid_observations = tiger_observation_enum.keys()

        # Define the acceptable reward values.
        self.valid_rewards = tiger_reward_enum.keys()

        # Set the accuracy of the listen action.
        if 'tiger-listen-accuracy' not in options:
            options["tiger-listen-accuracy"] = self.default_listen_accuracy
        # end if
        self.listen_accuracy = float(options["tiger-listen-accuracy"])

        # Make sure the accuracy value is valid.
        assert 0.0 <= self.listen_accuracy and self.listen_accuracy <= 1.0

        # Place the tiger.
        self.place_tiger()

        # Set an initial percept.
        self.observation = oNull
        self.reward = 0
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        if (action == aListen):
            # Listen for the tiger, and return the correct door with probability
            # equal to self.listen_accuracy.
            self.reward = rListen
            self.observation = self.tiger if random.random() < self.listen_accuracy else self.gold
        else:
            # Open a door. Set the reward according to what we find.
            if (action == aLeft):
                self.reward = rEaten if self.tiger == oLeft else rGold
            elif (action == aRight):
                self.reward = rEaten if self.tiger == oRight else rGold
            # end if

            # Set the observation to null, and replace the tiger and the gold.
            self.observation = oNull
            self.place_tiger()
        # end if

        return (self.observation, self.reward)
    # end def

    def place_tiger(self):
        """ Randomly places the tiger behind one door and gold behind the other.

            (Called `placeTiger` in the C++ version.)
        """

        self.tiger = oLeft if random.random() < 0.5 else oRight
        self.gold  = oRight if self.tiger == oLeft else oLeft
    # end def

    def print(self):
        """ Returns a string indicating the status of the environment.
        """

        action_text      = {aListen: "listen",
                            aLeft: "open left door",
                            aRight: "open right door"}
        observation_text = {oNull: "null",
                            oLeft: "hear tiger at left door",
                            oRight: "hear tiger at right door"}
        reward_text      = {rEaten: "eaten",
                            rListen: "listen",
                            rGold: "gold!"}

        # Show what just happened, correcting the reward for being defined relative to 100.
        message = "action = %s, observation = %s, reward = %s (%d)" % \
                  (action_text[self.action],
                   observation_text[self.observation],
                   reward_text[self.reward],
                   (self.reward - 100))

        return message
    # end def
# end class