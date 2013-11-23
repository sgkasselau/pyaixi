#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for Kuhn Poker: a simplified, zero-sum version of poker.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random

from pyaixi import environment, util

# Define a enumeration to represent agent interactions with the environment,
# such as betting or passing.
kuhn_poker_action_enum = util.enum('aBet', 'aPass')

# Define a enumeration to represent environment observations, such as card values,
# and opponent bet status
# The final observation is of the form `agent-card + opponent-bet-status`.
kuhn_poker_observation_enum = util.enum(oJack = 0, oQueen = 1, oKing = 2, oBet = 0, oPass = 4)

# Define a enumeration to represent rewards as a result of actions: betting and losing,
# betting and winning, passing and losing, passing and winning.
kuhn_poker_reward_enum = util.enum(rBetLoss = 0, rPassLoss = 1, rPassWin = 3, rBetWin = 4)

# Define some shorthand notation for ease of reference.
aBet   = kuhn_poker_action_enum.aBet
aPass  = kuhn_poker_action_enum.aPass

oJack  = kuhn_poker_observation_enum.oJack
oQueen = kuhn_poker_observation_enum.oQueen
oKing  = kuhn_poker_observation_enum.oKing
oBet   = kuhn_poker_observation_enum.oBet
oPass  = kuhn_poker_observation_enum.oPass

rBetLoss  = kuhn_poker_reward_enum.rBetLoss
rPassLoss = kuhn_poker_reward_enum.rPassLoss
rPassWin  = kuhn_poker_reward_enum.rPassWin
rBetWin   = kuhn_poker_reward_enum.rBetWin

class KuhnPoker(environment.Environment):
    """ Kuhn Poker is a simplified, zero-sum, two player poker variant that uses a
        deck of three cards: a King, Queen and Jack.

        While considerably less sophisticated than popular poker variants such as
        Texas Hold'em, well-known strategic concepts such as bluffing and slow-playing
        remain characteristic of strong play.

        In this setup, the agent acts second in a series of rounds. Two actions, pass
        (`aPass`) or bet (`aBet`), are available to each player.

        A bet action requires the player to put an extra chip into play. At the
        beginning of each round, each player puts a chip into play. The environment
        (opponent) then decides whether to pass or bet; betting will win the round
        if the agent subsequently passes, otherwise a showdown will occur.

        In a showdown, the player with the highest card wins the round (i.e. King beats
        Queen, Queen beats Jack). If the environment (opponent) passes, the agent
        can either bet or pass; passing leads immediately to a showdown, while
        betting requires the environment (opponent) to either bet to force a
        showdown, or to pass and let the agent win the round uncontested.

        The winner of the round gains a reward equal to the total chips in play
        (`rPassWin` or `rBetWin`), the loser receives a penalty equal to the number
        of chips they put into play this round (`rPassLoss` or `rBetLoss`).

        At the end of the round, all chips are removed from play and another round begins.

        Domain characteristics:

        - environment: "kuhn_poker"
        - maximum action: 1 (1 bit)
        - maximum observation: 6 (3 bits)
        - maximum reward: 4 (3 bits)
    """

    # Instance attributes.

    # Betting constants.
    bet_probability_king  = 0.7
    bet_probability_queen = (1.0 + bet_probability_king) / 3.0
    bet_probability_jack  = bet_probability_king / 3.0

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the Kuhn Poker environment from the given options.

             - `options` is a dictionary of named options and their values.
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Define the acceptable action values.
        self.valid_actions = kuhn_poker_action_enum.keys()

        # Define the acceptable observation values.
        self.valid_observations = kuhn_poker_observation_enum.keys()

        # Define the acceptable reward values.
        self.valid_rewards = kuhn_poker_reward_enum.keys()

        # Set the initial reward.
        self.reward = 0

        # Set up the game.
        self.reset()
    # end def

    def card_to_string(self, card):
        """ Returns a human-readable string version of the given card observation.

            (Called `cardToString` in the C++ version.)
        """

        if card == oJack:
            return "jack"
        elif card == oQueen:
            return "queen"
        elif card == oKing:
            return "king"
        else:
            return ""
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.
            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # If the agent did not call the environments bet then the agent loses
        if self.action == aPass and self.env_action == aBet:
            self.reward = rPassLoss
            self.reset()
            return (self.observation, self.reward)
        # end if

        # If the environment passed and the agent bet, then the environment has
        # a chance to change its mind.
        if self.action == aBet and self.env_action == aPass:
            if self.env_card == oQueen and random.random() < self.bet_probability_queen:
                # Bet with the internal-default probability on seeing a king while having a queen.
                self.env_action = aBet
            elif self.env_card == oKing:
                # The environment always bets if it has a king.
                self.env_action = aBet
            else:
                # Environment continues to pass, so agent wins.
                self.reward = rPassWin
                self.reset()
                return (self.observation, self.reward)
            # end if
        # end if

        # Players have bet the same amount, winner has highest card.
        agent_wins = (self.env_card == oJack) or \
                     (self.env_card == oQueen and self.agent_card == oKing)

        if agent_wins:
            self.reward = rBetWin if self.env_action == aBet else rPassWin
        else:
            self.reward = rBetLoss if self.action == aBet else rPassLoss
        # end if
        self.reset()

        return (self.observation, self.reward)
    # end def

    def print(self):
        """ Returns a string indicating the status of the environment.
        """

        # Add cards and bets to the output.
        message = "agent card = %s" % self.card_to_string(self.agent_previous_card) + \
                  ", environment card = %s" % self.card_to_string(self.env_previous_card) + \
                  ", agent %s" % ("passes" if self.action == aPass else "bets") + \
                  ", environment %s" % ("passes" if self.env_previous_action == aPass else "bets") + \
                  os.linesep

        # Add the winner and the reward to the output.
        agent_wins = (self.reward == rPassWin or self.reward == rBetWin)
        message += "agent %s" % ("wins" if agent_wins else "loses") + \
                   ", reward = %d (%d)" % (self.reward, self.reward - 2) + os.linesep

        return message
    # end def

    def random_card(self):
        """ Returns a card uniformly at random.

            (Called `randomCard` in the C++ version.)
        """

        return random.choice([oJack, oQueen, oKing])
    # end def

    def reset(self):
        """ Resets the environment by randomly placing the tiger behind one door,
            the gold behind the other, and reseating the agent.
        """

        # Save the previous actions/cards for use by print().
        self.env_previous_action = getattr(self, 'env_action', None)
        self.agent_previous_card = getattr(self, 'agent_card', None)
        self.env_previous_card   = getattr(self, 'env_card', None)

        # Deal cards.
        self.agent_card = self.random_card()
        self.env_card = self.agent_card
        while (self.env_card == self.agent_card):
            self.env_card = self.random_card()
        # end def

        # Choose the environment's first action. Bet with a certain probability
        # on jack and king, pass on queen.
        if (self.env_card == oJack):
            self.env_action = aBet if (random.random() < self.bet_probability_jack) else aPass
        elif(self.env_card == oQueen):
            # Always pass on a Queen.
            self.env_action = aPass
        elif(self.env_card == oKing):
            self.env_action = aBet if (random.random() < self.bet_probability_king) else aPass
        # end if

        # Compute an observation: agent-card + environment-bet-status
        self.observation = self.agent_card + (oPass if (self.env_action == aPass) else oBet)
    # end def
# end class