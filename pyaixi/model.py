#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines a base class for models used by agents to make predictions.
"""

from __future__ import division
from __future__ import unicode_literals

import copy
import random

from pyaixi import util

class Model:
    """ This base class represents the minimum class elements for an AIXI-compatible model.

        The following attributes and methods must be available in all models, in order
        for the agent to update the model with percepts and make predictions.

        - `clear` resets the model to its initial state.

        - `update(symbol_or_list_of_symbols)` updates the model and the history (if applicable)
          after the agent has observed new percepts.

        - `update_history(symbol_or_list_of_symbols)` updates just the history (if applicable)
          after the agent has executed an action.

        - `revert()` reverts the last update to the model.

        - `revert_history()` deletes the recent history (if applicable).

        - `predict()` predicts the probability of future outcomes.

        - `generate_random_symbols_and_update()` samples a sequence from the
           model, updating the model with each symbol as it is sampled.

        - `generate_random_symbols()` samples a sequence of a specified length,
           updating the model with each symbol as it is sampled, without updating the model (possibly by
           reverting the update), so that the model is in the same state as it was before the
           sampling.

        - `size` gives an indication (if possible) of the size of the model.
    """

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct an AIXI-compatible model from the given configuration values.

             - `options` is a dictionary of named options and their values.
        """

        # Stores the given configuration options.
        # (Called `m_options` in the C++ version.)
        self.options = options

        # Read off the maximum and minimum action, observation, and reward values, for use in generating these.
        assert 'min-action' in options, \
               "The minimum action wasn't successfully passed to the model class."
        self.minimum_action = options['min-action']

        assert 'max-action' in options, \
               "The maximum action wasn't successfully passed to the model class."
        self.maximum_action = options['max-action']

        assert 'min-observation' in options, \
               "The minimum observation wasn't successfully passed to the model class."
        self.minimum_observation = options['min-observation']

        assert 'max-observation' in options, \
               "The maximum observation wasn't successfully passed to the model class."
        self.maximum_observation = options['max-observation']

        assert 'min-reward' in options, \
               "The minimum reward wasn't successfully passed to the model class."
        self.minimum_reward = options['min-reward']

        assert 'max-reward' in options, \
               "The maximum reward wasn't successfully passed to the model class."
        self.maximum_reward = options['max-reward']
    # end def

    def clear(self):
        """ Resets the model to its initial state.

            WARNING: this method should be overriden by inheriting classes.
        """
        pass
    # end def

    def size(self):
        """ Returns the size of the model.

            WARNING: this method should be overriden by inheriting classes.
        """
        return 0
    # end def

    def generate_random_symbols(self, symbol_count):
        """ Returns a symbol string of a specified length by sampling from the model.

            - `symbol_count`: the number of symbols to generate.

            WARNING: this method should be overriden by inheriting classes.
        """
        return []
    # end def

    def generate_random_symbols_and_update(self, symbol_count):
        """ Returns a specified number of random symbols distributed according to
            the model, updating the model with the newly
            generated symbols.

            - `symbol_count`: the number of symbols to generate.

            WARNING: this method should be overriden by inheriting classes.
        """
        return []
    # end def

    def predict(self, symbol_list):
        """ Returns the conditional probability of a symbol (or a list of symbols), considering the model's history.

            - `symbol_list` The symbol (or list of symbols) to estimate the conditional probability of.

            WARNING: this method should be overriden by inheriting classes.
        """
        return 0
    # end def

    def revert(self, symbol_count = 1):
        """ Restores the model to its state prior to a specified number of updates.
     
            - `symbol_count`: the number of updates (symbols) to revert. (Default of 1.)

            WARNING: this method should be overriden by inheriting classes.
        """
        pass
    # end def

    def revert_history(self, symbol_count = 1):
        """ Shrinks the model history without affecting the model.

            WARNING: this method should be overriden by inheriting classes.
        """
        pass
    # end def

    def size(self):
        """ Returns the size of the model.

            WARNING: this method should be overriden by inheriting classes.
        """
        return 0
    # end def

    def update(self, symbol_list):
        """ Updates the model with a new symbol, or a list of symbols.

            - `symbol_list`: the symbol (or list of symbols) with which to update the model.
                              (The model is updated with symbols in the order they appear in the list.)

            WARNING: this method should be overriden by inheriting classes.
        """
        pass
    # end def

    def update_history(self, symbol_list):
        """ Appends a symbol (or a list of symbols) to the model's history without updating the model.

            - `symbol_list`: the symbol (or list of symbols) to add to the history.

            WARNING: this method should be overriden by inheriting classes.
        """
        pass
    # end def
# end class