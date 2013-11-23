#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for an extended version of an agent interacting with an environment
where there's a tiger and a pot of gold hidden separately, behind two closed doors.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from pyaixi import environment, util

# Define a enumeration to represent agent interactions with the environment,
# such as listening, just standing, or opening doors.
extended_tiger_action_enum = util.enum('aListen', 'aLeft', 'aRight', 'aStand')

# Define a enumeration to represent environment observations: either not
# hearing the tiger, or hearing it from behind either door.
extended_tiger_observation_enum = util.enum('oNull', 'oLeft', 'oRight')

# Define a enumeration to represent rewards as a result of actions: being eaten by the tiger,
# performing an invalid action (listening while sitting), getting information from listening,
# just standing there, or finding the gold.
# NOTE: since the enumeration values need to be positive, these values are defined relative to
#       100.
#       So -100 points is 0, -1 points is 99, and 30 points is 130.
extended_tiger_reward_enum = util.enum(rInvalid = 0, rTiger = 0, rStand = 99, rListen = 100, rGold = 130)

# Define some shorthand notation for ease of reference.
aListen = extended_tiger_action_enum.aListen
aLeft   = extended_tiger_action_enum.aLeft
aRight  = extended_tiger_action_enum.aRight
aStand  = extended_tiger_action_enum.aStand

oNull  = extended_tiger_observation_enum.oNull
oLeft  = extended_tiger_observation_enum.oLeft
oRight = extended_tiger_observation_enum.oRight

rInvalid  = extended_tiger_reward_enum.rInvalid
rTiger = extended_tiger_reward_enum.rTiger
rStand = extended_tiger_reward_enum.rStand
rListen = extended_tiger_reward_enum.rListen
rGold   = extended_tiger_reward_enum.rGold

class ExtendedTiger(environment.Environment):
    """ The environment is a more elaborate version of Tiger.

        There are two doors and a stool. A tiger is hidden behind one door
        and a pot of gold is hidden behind the other.

        The agent begins each round sitting on the stool where it
        may either listen for the tiger (`aListen`) or stand up
        `aStand`.

        Listening for the tiger results in an observation
        which correctly describes the tiger's whereabouts with probability
        `m_listen_accuracy` and a reward of `rListen`.

        Standing up results in an uninformative observation (`oNull`) and
        a reward of `rStand`.

        Once the agent is standing, it may open either the left or right door
        (`oLeft` and `oRight`). Doing so results in an uninformative observation
        `oNull` and a reward based on what is behind the door (`rGold` or `rTiger`).

        After opening a door the agent is re-seated and the tiger and gold randomly
        re-allocated to a door (`reset()`).

        Attempting to open a door while seated, to listen while standing, or to
        stand while already standing will result in an uninformative observation
        (`Null`) and a reward of `rInvalid`.

        Domain characteristics:

        - environment: "extended_tiger"
        - maximum action: 3 (2 bits)
        - maximum observation: 2 (2 bits)
        - maximum reward: 130 (8 bits)

        Configuration options:
        - `tiger-listen-accuracy`: the probability that listening (while seated) for the
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
        self.valid_actions = extended_tiger_action_enum.keys()

        # Define the acceptable observation values.
        self.valid_observations = extended_tiger_observation_enum.keys()

        # Define the acceptable reward values.
        self.valid_rewards = extended_tiger_reward_enum.keys()

        # Set the accuracy of the listen action.
        if 'tiger-listen-accuracy' not in options:
            options["tiger-listen-accuracy"] = self.default_listen_accuracy
        # end if
        self.listen_accuracy = float(options["tiger-listen-accuracy"])

        # Make sure the accuracy value is valid.
        assert 0.0 <= self.listen_accuracy and self.listen_accuracy <= 1.0

        # Set an initial percept.
        self.observation = oNull
        self.reward = 0

        # Place the tiger, and the gold.
        self.reset()
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Unless explicitly accounted for, the action is invalid, and the observation is null.
        self.observation = oNull
        self.reward = rInvalid

        if (action == aListen and self.sitting):
            # Listen while sitting down, and return the correct door with probability
            # equal to self.listen_acurracy.
            self.observation = self.tiger if random.random() < self.listen_accuracy else self.gold
            self.reward = rListen
        elif (action == aLeft and not self.sitting):
            # Open the left door while standing. Get a reward based on what was behind
            # the door. Reseat the agent and reallocate the tiger and the gold.
            self.reward = rTiger if self.tiger == oLeft else rGold
            self.reset()
        elif (action == aRight and not self.sitting):
            # Open the right door while standing. Get a reward based on what was behind
            # the door. Reseat the agent and reallocate the tiger and the gold.
            self.reward = rTiger if self.tiger == oRight else rGold
            self.reset()
        elif (action == aStand and self.sitting):
            # Stand from a seated position. Get the reward for standing.
            self.reward = rStand
            self.sitting = False
        # end if

        return (self.observation, self.reward)
    # end def

    def print(self):
        """ Returns a string indicating the status of the environment.
        """

        action_text      = {aListen: "listen",
                            aLeft: "open left door",
                            aRight: "open right door",
                            aStand: "stand up"}
        observation_text = {oNull: "null",
                            oLeft: "hear tiger at left door",
                            oRight: "hear tiger at right door"}
        reward_text      = {rTiger: "eaten",
                            rInvalid: "invalid action",
                            rStand: "stand up",
                            rListen: "listen",
                            rGold: "gold!"}
        state_text       = {False: "standing",
                            True: "sitting"}

        # Show what just happened, correcting the reward for being defined relative to 100.
        message = "action = %s, observation = %s, reward = %s (%d), agent is now %s"% \
                  (action_text[self.action],
                   observation_text[self.observation],
                   reward_text[self.reward],
                   (self.reward - 100),
                   state_text[self.sitting])

        return message
    # end def

    def reset(self):
        """ Resets the environment by randomly placing the tiger behind one door,
            the gold behind the other, and reseating the agent.
        """

        # Place the tiger randomly.
        self.tiger = oLeft if random.random() < 0.5 else oRight

        # Place the gold behind the opposite door.
        self.gold  = oRight if self.tiger == oLeft else oLeft

        # Start the agent sitting down.
        self.sitting = True;
    # end def
# end class