#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define a class to implement a Monte Carlo search tree.
"""

from __future__ import division
from __future__ import unicode_literals

import os
import math
import random
import sys

# Insert the package's parent directory into the system search path, so that this package can be
# imported when the aixi.py script is run directly from a release archive.
PROJECT_ROOT = os.path.realpath(os.path.join(os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from pyaixi import util

# An enumeration type used to specify the type of Monte Carlo search node.
# Chance nodes represent a set of possible observation
# (one child per observation) while decision nodes
# represent sets of possible actions (one child per action).
# Decision and chance nodes alternate.
nodetype_enum = util.enum('chance', 'decision')

# Define some short cuts for ease of reference.
chance_node = nodetype_enum.chance
decision_node = nodetype_enum.decision

class MonteCarloSearchNode:
    """ A class to represent a node in the Monte Carlo search tree.
        The nodes in the search tree represent simulated actions and percepts
        between an agent following an upper confidence bounds (UCB) policy and a generative
        model of the environment represented by a context tree.

        The purpose of the tree is to determine the expected reward of the
        available actions through sampling. Sampling proceeds several time steps
        into the future according to the size of the agent's horizon.
        (`MC_AIXI_CTW_Agent.horizon`)
 
        The nodes are one of two types (`nodetype_enum`), decision nodes are those
        whose children represent actions from the agent and chance nodes are those
        whose children represent percepts from the environment.

        Each MonteCarloSearchNode maintains several bits of information:

          - The current value of the sampled expected reward
            (`MonteCarloSearchNode.mean`, `MonteCarloSearchNode.expectation`).

          - The number of times the node has been visited during the sampling
            (`MonteCarloSearchNode.visits`).

          - The type of the node (MonteCarloSearchNode.type).

          - The children of the node (`MonteCarloSearchNode.children`).
            The children are stored in a dictionary indexed by action (if
            it is a decision node) or percept (if it is a chance node).

        The `MonteCarloSearchNode.sample` method is used to sample from the current node and
        the `MonteCarloSearchNode.selectAction` method is used to select an action according
        to the UCB policy.
    """

    # Static list of all attributes. (Defining slots helps save memory.)

    __slots__ = ['action_search_limit', 'exploration_constant', 'unexplored_bias', 'children', 'mean', 'type', 'visits']

    # Instance methods.

    def __init__(self, nodetype, options = {}):
        """ Create a new search node of the given type.
        """

        # Set some values from the options, if they exist there, else use some defaults.

        # Maximum number of actions to generate and evaluate while searching, in lieu of a
        # set of valid actions being provided by the environment.
        # (Useful in settings where the action range is very large.)
        self.action_search_limit = options.get('mc-action-search-limit', 10)

        # Exploration constant for the UCB action policy.
        self.exploration_constant = options.get('mc-exploration-constant', 2.0)

        # Unexplored action bias.
        self.unexplored_bias = options.get('mc-unexplored-bias', 1000000000.0)

        # The children of this node.
        # The symbols used as keys at each level may be either action or observation,
        # depending on what type of node this is.
        self.children = {}

        # The sampled expected reward of this node.
        # (Called `m_mean` and `expectation()` in the C++ version.)
        self.mean = 0.0

        # The type of this node indicates whether its children represent actions
        # (decision node) or percepts (chance node).
        # (Called `m_type` in the C++ version.)
        assert nodetype in nodetype_enum, "The given value %s is a not a valid node type." % str(nodetype)
        self.type = nodetype

        # The number of times this node has been visited during sampling.
        # (Called `m_visits` in the C++ version.)
        self.visits = 0
    # end def

    def sample(self, agent, horizon):
        """ Returns the accumulated reward from performing a single sample on this node.

            - `agent`: the agent doing the sampling

            - `horizon`: how many cycles into the future to sample
        """

        # Set an initial reward.
        reward = 0.0

        # Do we need to continue sampling, or use a playout policy?
        # Have we already reached the given horizon or the maximum search depth?
        if (horizon == 0):
            # Yes, we've reached the maximum search depth.
            # Return the initial reward.
            return reward
        elif (self.type == chance_node):
            # We're at a chance node, so we need to continue sampling.

            # Generate a percept at random using the agent's environment model,
            # and continue sampling.
            observation, random_reward = agent.generate_percept_and_update()

            # If this observation is new to this node, add it as a decision observation child node.
            if observation not in self.children:
                self.children[observation] = MonteCarloSearchNode(decision_node)
            # end def
            observation_child = self.children[observation]

            # Get the reward for this observation by continuing the search recursively onto the child.
            reward = random_reward + observation_child.sample(agent, horizon - 1)
        elif (self.visits == 0):
            # We are at an (unvisited) decision node.
            # Either the node is previously unvisited, or we have exceeded the maximum tree depth.
            # Either way, use the playout policy to estimate the future reward.
            reward = agent.playout(horizon)
        else:
            # We are at a previously-visited decision node.
            # Choose an action according to the UCB policy and continue sampling.
            action = self.select_action(agent)
            agent.model_update_action(action)

            # If this action is new to this node, add it as a chance action child node.
            # Do we have a child corresponding to this action?
            if action not in self.children:
                # No. Create one.
                self.children[action] = MonteCarloSearchNode(chance_node)
            # end def
            action_child = self.children[action]

            # Get the reward for this action by continuing the search recursively onto the child.
            # NOTE: should this be horizon - 1, like observation_child_sample above?
            #       (The C++ version is just the horizon as well.)
            reward = action_child.sample(agent, horizon)
        # end def

        # Update the expected reward and number of visits to the current node.
        visits = float(self.visits)
        self.mean = (reward + (visits * self.mean)) / (visits + 1.0)
        self.visits += 1

        # Return the calculated reward.
        return reward
    # end def

    def select_action(self, agent):
        """ Returns an action selected according to UCB policy.

             - `agent`: the agent which is doing the sampling.

            (Called `selectAction` in the C++ version.)
        """

        explore_bias = float(agent.horizon * agent.maximum_reward())
        exploration_numerator = (self.exploration_constant * math.log(self.visits))

        # Compute the best action according to the UCB formula.
        best_action = None
        best_priority = float("-inf")

        # Determine which set of actions to use, depending on whether we've got a set of valid actions to use.
        if hasattr(agent.environment, 'valid_actions'):
            # We've got a set of valid actions, so use that.
            action_range = agent.environment.valid_actions
        else:
            # We don't have a set of valid actions, only (presumably) a range of values to use.

            # Can we access the agent to generate an action to evaluate?
            if hasattr(agent, 'generate_action'):
                # Yes, so try to sample some random actions, up to the action search limit.
                action_range = [agent.generate_action() for i in xrange(0, self.action_search_limit)]
            else:
                # No, so pick some actions in the range we have.

                # Is this range larger than the action search limit?
                minimum_action = agent.environment.minimum_action()
                maximum_action = agent.environment.maximum_action()
                if (maximum_action - minimum_action) > self.action_search_limit:
                    # Yes, so just choose some actions sampled at random in the range from the minimum action value,
                    # through to the maximum action value, inclusive.
                    action_range = random.sample(xrange(minimum_action, maximum_action + 1), self.action_search_limit)
                else:
                    # No, so just use the range of the minimum action value through to maximum action value, inclusive.
                    action_range = xrange(minimum_action, maximum_action + 1)
                # end if
            # end if
        # end if

        for action in action_range:
            # Find any children nodes related to this action.
            node = self.children.get(action, None)

            # Use the UCB formula to determine priority of this node.
            priority = 0.0
            if (node is None or node.visits == 0):
                # This is a previously unexplored node.
                # Give it the unexplored bias.
                priority = self.unexplored_bias
            else:
                # This is a previously explored node.
                priority = node.mean + (explore_bias * math.sqrt(exploration_numerator / node.visits))
            # end def

            # Update the best action if necessary, breaking ties randomly.
            if (priority > (best_priority + (random.random() * 0.001))):
                best_action = action
                best_priority = priority
            # end if
        # end for

        return best_action
    # end def
# end class