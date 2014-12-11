#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines a class for the MC-AIXI-CTW agent.
"""

from __future__ import division
from __future__ import unicode_literals

import copy
import os
import random
import sys

# Insert the package's parent directory into the system search path, so that this package can be
# imported when the aixi.py script is run directly from a release archive.
PROJECT_ROOT = os.path.realpath(os.path.join(os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Ensure xrange is defined on Python 3.
from six.moves import xrange

from pyaixi.agents import mc_aixi
from pyaixi.prediction import ctw_context_tree


class MC_AIXI_CTW_Agent(mc_aixi.MC_AIXI_Agent):
    """ This class represents a MC-AIXI-CTW agent, inheriting methods from the more abstract
        Monte Carlo-AIXI agent.

        It includes the high-level logic for choosing suitable actions.
        In particular, the agent maintains an internal model of the environment using
        a context tree.

        It uses this internal model to to predict the probability of future outcomes:

         - `get_predicted_action_probability()`
         - `percept_probability()`

        as well as to generate actions and precepts according to the model distribution:

         - `generate_action()`
         - `gen_percept()`
         - `generate_percept_and_update()`
         - `generate_random_action()`

        Actions are chosen via the UCT algorithm, which is orchestrated by a
        high-level search function and a playout policy:

         - `search()`
         - `playout()`
         - `horizon`
         - `mc_simulations`
         - `search_tree`

        Several functions decode/encode actions and percepts between the
        corresponding types (i.e. `action_enum`, `percept_enum`) and generic
        representation by symbol lists:

         - `decode_action()`
         - `decode_observation()`
         - `decode_percept()`
         - `decode_reward()`
         - `encode_action()`
         - `encode_percept()`

        There are various attributes which describe the agent and its
        interaction with the environment so far:

         - `age`
         - `average_reward`
         - `history_size()`
         - `horizon`
         - `last_update`
         - `maximum_action()`
         - `maximum_bits_needed()`
         - `maximum_reward()`
         - `total_reward`
    """

    # Instance methods.

    def __init__(self, environment = None, options = {}):
        """ Construct a MC-AIXI-CTW learning agent from the given configuration values and the environment.

             - `environment` is an instance of the pyaixi.Environment class that the agent with interact with.
             - `options` is a dictionary of named options and their values.

            `options` must contain the following mandatory options:
             - `agent-horizon`: the agent's planning horizon.
             - `ct-depth`: the depth of the context tree for this agent, in symbols/bits.
             - `mc-simulations`: the number of simulations to run when choosing new actions.

            The following options are optional:
             - `learning-period`: the number of cycles the agent should learn for.
                                  Defaults to '0', which is indefinite learning.
        """

        # The agent's context tree depth.
        # Retrieved from the given options under 'ct-depth'. Mandatory.
        # (Called `ct_depth` in the C++ version.)
        assert 'ct-depth' in options, \
               "The required 'ct-depth' context tree depth option is missing from the given options."
        self.depth = int(options['ct-depth'])

        # (CTW) Context tree representing the agent's model of the environment.
        # Created for this instance.
        # (Called `m_ct` in the C++ version.)
        context_tree = ctw_context_tree.CTWContextTree(self.depth, options = options)

        # Set up the base agent options, which handles getting and setting the learning period, amongst other basic values,
        # passing in the created context tree as a model.
        mc_aixi.MC_AIXI_Agent.__init__(self, environment = environment, model = context_tree, options = options)
    # end def
# end class