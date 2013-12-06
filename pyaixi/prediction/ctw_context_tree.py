#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define classes to implement context trees according to the Context Tree Weighting algorithm.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import random

# The value ln(0.5).
# This value is used often in computations and so is made a constant for efficiency reasons.
log_half = math.log(0.5)

class CTWContextTreeNode:
    """ The CTWContextTreeNode class represents a node in an action-conditional context tree.


        The purpose of each node is to calculate the weighted probability of observing
        a particular bit sequence.

        In particular, denote by `n` the current node, by `n0` and `n1`  the child nodes,
        by `h_n` the subsequence of the history relevant to node `n`, and by `a`
        and `b` the number of zeros and ones in `h_n`.

        Then the weighted block probability of observing `h_n` at node `n` is given by

          P_w^n(h_n) :=
  
            Pr_kt(h_n)                        (if n is a leaf node)
            1/2 Pr_kt(h_n) +
            1/2 P_w^n0(h_n0) P_w^n1(h_n1)     (otherwise)

        where `Pr_kt(h_n) = Pr_kt(a, b)` is the Krichevsky-Trofimov (KT) estimator defined by the relations

          Pr_kt(a + 1, b) = (a + 1/2)/(a + b + 1) Pr_kt(a, b)

          Pr_kt(a, b + 1) = (b + 1/2)/(a + b + 1) Pr_kt(a, b)

        and the base case

          Pr_kt(0, 0) := 1


        In both relations, the fraction is referred to as the update multiplier and corresponds to the
        probability of observing a zero (first relation) or a one (second relation) given we have seen
        `a` zeros and `b` ones.

        Due to numerical issues, the implementation uses logarithmic probabilities
        `log(P_w^n(h_n))` and `log(Pr_kt(h_n)` rather than normal probabilities.

        These probabilities are recalculated during updates (`update()`)
        and reversions (`revert()`) to the context tree that involves the node.

        - The KT estimate is accessed and stored using `log_kt`.
          It is updated from the previous estimate by multiplying with the update multiplier as
          calculated by `log_kt_multiplier()`.

        - The weighted probability is access and stored using `log_probability`.
          It is recalculated by `update_log_probability()`.

        In order to calculate these probabilities, `CTWContextTreeNode` also stores:

        - Links to child nodes: `children`

        - The number of symbols (zeros and ones) in the history subsequence relevant to the
          node: `symbol_count`.


        The `CTWContextTreeNode` class is tightly coupled with the `ContextTree` class.

        Briefly, the `ContextTree` class:

        - Creates and deletes nodes.

        - Tells the appropriate nodes to update/revert their probability estimates.

        - Samples actions and percepts from the probability distribution specified
          by the nodes.
    """

    # Instance methods.

    def __init__(self, tree = None):
        """ Construct a node of the context tree.
        """

        # The children of this node.
        self.children = {}

        # The tree object associated with this node.
        self.tree = tree

        # The cached KT estimate of the block log probability for this node.
        # This value is computed only when the node is changed by the update or revert methods.
        # (Called `m_log_kt` in the C++ version.)
        self.log_kt = 0.0

        # The cached weighted log probability for this node.
        # This value is computed only when the node is changed by the update or revert methods.
        # (Called `m_log_probability` and accessed via `logBlockProbability` in the C++ version.)
        self.log_probability = 0.0

        # The count of the symbols in the history subsequence relevant to this node.
        # (Called `m_count` in the C++ version.)
        self.symbol_count = {0: 0, 1: 0}
    # end def

    def is_leaf_node(self):
        """ Return True if the node is a leaf node, False otherwise.
        """

        # If this node has no children, it's a leaf node.
        return self.children == {}
    # end def

    def log_kt_multiplier(self, symbol):
        """ Returns the logarithm of the KT-estimator update multiplier.

           The log KT estimate of the conditional probability of observing a zero given
           we have observed `a` zeros and `b` ones at the current node is

             log(Pr_kt(0 | 0^a 1^b)) = log((a + 1/2)/({a + b + 1))

           Similarly, the estimate of the conditional probability of observing a one is

             log(\Pr_kt(1 |0^a 1^b)) = log((b + 1/2)/(a + b + 1))

           - `symbol`: the symbol for which to calculate the log KT estimate of
             conditional probability.

             0 corresponds to calculating `log(Pr_kt(0 | 0^a 1^b)` and
             1 corresponds to calculating `log(Pr_kt(1 | 0^a 1^b)`.
        """

        numerator = self.symbol_count[symbol] + 0.5
        denominator = self.visits() + 1

        return math.log(numerator / denominator)
    # end def

    def revert(self, symbol):
        """ Reverts the node to its state immediately prior to the last update.
            This involves updating the symbol counts, recalculating the cached
            probabilities, and deleting unnecessary child nodes.

            - `symbol`: the symbol used in the previous update.
        """

        # Decrease the count for this symbol.
        symbol = int(symbol)
        this_symbol_count = self.symbol_count[symbol]
        if this_symbol_count > 1:
            self.symbol_count[symbol] = this_symbol_count - 1
        else:
            self.symbol_count[symbol] = 0
        # end if

        # If the number of visits to the child associated with this symbol is now zero,
        # this child is now redundant, and should be removed.
        redundant_child = self.children.get(symbol, None)
        if redundant_child is not None and (redundant_child.visits() == 0):
            # Decrease the tree size by the size of the children of the redundant child.
            self.tree.tree_size -= redundant_child.size()

            del self.children[symbol]
            del redundant_child
        # end if

        # Revert the KT estimate.
        self.log_kt -= self.log_kt_multiplier(symbol)

        # Update the weighted probability.
        self.update_log_probability()
    # end def

    def size(self):
        """ The number of descendants of this node.
        """

        # Iterate over the direct children of this node, collecting the size of each sub-tree.
        return 1 + sum([child.size() for child in self.children.values()])
    # end def

    def update(self, symbol):
        """ Updates the node after having observed a new symbol.
            This involves updating the symbol counts and recalculating the cached probabilities.

            - `symbol`: the symbol that was observed.
        """

        # Update the KT estimate.
        self.log_kt += self.log_kt_multiplier(symbol)

        # Update the weighted probability.
        self.update_log_probability()

        # Update the count for this symbol.
        self.symbol_count[symbol] += 1
    # end def

    def update_log_probability(self):
        """ This method calculates the logarithm of the weighted probability for this node.

            Assumes that `log_kt` and `log_probability` is correct for each child node.

              log(P^n_w) :=
                  log(Pr_kt(h_n)            (if n is a leaf node)
                  log(1/2 Pr_kt(h_n)) + 1/2 P^n0_w x P^n1_w)
                                            (otherwise)
            and stores the value in log_probability.
     
            Because of numerical issues, the implementation works directly with the
            log probabilities `log(Pr_kt(h_n)`, `log(P^n0_w)`,
            and `log(P^n1_w)` rather than the normal probabilities.

            To compute the second case of the weighted probability, we use the identity

                log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))       a,b > 0

            to rearrange so that logarithms act directly on the probabilities:

                log(1/2 Pr_kt(h_n) + 1/2 P^n0_w P^n1_w) =

                    log(1/2) + log(Pr_kt(h_n))
                      + log(1 + exp(log(P^n0_w) + log(P^n1_w)
                                    - log(Pr_kt(h_n)))

                    log(1/2) + log(P^n0_w) + log(P^n1_w)
                      + log(1 + exp(log(Pr_kt(h_n)
                                           - log(P^n0_w) + log(P^n1_w)))

            In order to avoid overflow problems, we choose the formulation for which
            the argument of the exponent `exp(log(b) - log(a))` is as small as possible.
        """

        # Calculate the log weighted probability.
        # If the current node is a leaf node (i.e. it has no children), this is just the KT estimate.
        # Otherwise, it is an even mixture of the KT estimate, and the product of the
        # weighted probabilities of the children.
        if self.children == {}:
            self.log_probability = self.log_kt
        else:
            # Calculate the sum of the log weighted probabilities of the child nodes.
            log_child_probability = math.fsum([child.log_probability for child in self.children.values()])

            # Calculate the log weighted probability.
            # Use the formulation which has the least chance of overflow.

            # Set 'a' to be the maximum of log_kt and log_child_probability, and 'b' to be the minimum.
            if self.log_kt >= log_child_probability:
                a = self.log_kt
                b = log_child_probability
            else:
                a = log_child_probability
                b = self.log_kt
            # end if

            # Use Python's log1p function to perform `log(1.0 + exp(b - a))`.
            self.log_probability = log_half + a + math.log1p(math.exp(b - a))
        # end if
    # end def

    def visits(self):
        """ Returns the number of times this context has been visited.
            This is the sum of the visits of the (immediate) child nodes.
        """

        return self.symbol_count[0] + self.symbol_count[1]
    # end def
# end class


class CTWContextTree:
    """ The high-level interface to an action-conditional context tree.
        Most of the mathematical details are implemented in the CTWContextTreeNode class, which is used to
        represent the nodes of the tree.
        CTWContextTree stores a reference to the root node of the tree (`root`), the history of
        updates to the tree (`history`), and the maximum depth of the tree (`depth`).

        It is primarily concerned with calling the appropriate functions in the appropriate nodes
        in order to deliver certain functionality:

        - `update(symbol_or_list_of_symbols)` updates the tree and the history
          after the agent has observed new percepts.

        - `update_history(symbol_or_list_of_symbols)` updates just the history
          after the agent has executed an action.

        - `revert()` reverts the last update to the tree.

        - `revert_history()` deletes the recent history.

        - `predict()` predicts the probability of future outcomes.

        - `generate_random_symbols_and_update()` samples a sequence from the
           context tree, updating the tree with each symbol as it is sampled.

        - `generate_random_symbols()` samples a sequence of a specified length,
           updating the tree with each symbol as it is sampled, then reverting all the
           updates so that the tree is in the same state as it was before the
           sampling.
    """

    def __init__(self, depth):
        """ Create a context tree of specified maximum depth.
            Nodes are created as needed.

            - `depth`: the maximum depth of the context tree.
        """

        # An list used to hold the nodes in the context tree that correspond to the current context.
        # It is important to ensure that `update_context()` is called before accessing the contents
        # of this list as they may otherwise be inaccurate.
        self.context = []

        # The maximum depth of the context tree.
        # (Called `m_depth` in the C++ version.)
        assert depth >= 0, "The given tree depth must be greater than zero."
        self.depth = depth

        # The history (a list) of symbols seen by the tree.
        # (Called `m_history` in the C++ version.)
        self.history = []

        # The root node of the context tree.
        # (Called `m_root` in the C++ version.)
        self.root = CTWContextTreeNode(tree = self)

        # The size of this tree.
        self.tree_size = 1
    # end def

    def clear(self):
        """ Clears the entire context tree including all nodes and history.
        """

        # Reset the history.
        self.history = []

        # Set a new root object, and reset the tree size.
        self.root.tree = None
        del self.root
        self.root = CTWContextTreeNode(tree = self)
        self.tree_size = 1

        # Reset the context.
        self.context = []
    # end def

    def generate_random_symbols(self, symbol_count):
        """ Returns a symbol string of a specified length by sampling from the context tree.

            - `symbol_count`: the number of symbols to generate.

            (Called `genRandomSymbols` in the C++ version.)
        """
        symbol_list = self.generate_random_symbols_and_update(symbol_list, symbol_count)
        self.revert(symbol_count)

        return symbol_list
    # end def

    def generate_random_symbols_and_update(self, symbol_count):
        """ Returns a specified number of random symbols distributed according to
            the context tree statistics and update the context tree with the newly
            generated symbols.

            - `symbol_count`: the number of symbols to generate.

            (Called `genRandomSymbolsAndUpdate` in the C++ version.)
        """

        symbol_list = []
        for i in xrange(0, symbol_count):
            # Pick either 0 or 1 based on the probability of the symbol 1 occuring in the context tree.
            symbol = 1 if (random.random() < self.predict(1)) else 0
            symbol_list += [symbol]
            self.update(symbol)
        # end for

        return symbol_list
    # end def

    def predict(self, symbol_list):
        """ Returns the conditional probability of a symbol (or a list of symbols), considering the history.

            Given a history sequence `h` and a symbol `y`, the estimated probability is given by

              rho(y | h) = rho(hy)/rho(h)

            where `rho(h) = P_w^epsilon(h)` is the weighted probability estimate of observing `h`
            evaluated at the root node `epsilon` of the context tree.

            - `symbol_list` The symbol (or list of symbols) to estimate the conditional probability of.
                            0 corresponds to `rho(0 | h)` and 1 to `rho(1 | h)`.
        """

        # Ensure that we have a list, by making this a list if it's a single symbol.
        if type(symbol_list) != list:
            symbol_list = [symbol_list]
        # end if


        # If there is insufficient context for a prediction, return the uniform
        # prediction 0.5 ^ length.
        symbol_list_length = len(symbol_list)
        if ((len(self.history) + symbol_list_length) <= self.depth):
            return 0.5 if symbol_list_length == 1 else math.pow(0.5, symbol_list_length)
        # end if

        # Calculate the probability of the symbol s given the history h using
        # p(s | h) = p(hs) / p(h) = exp(ln p(hs) - ln p(h)).
        prob_history = self.root.log_probability
        self.update(symbol_list)
        prob_sequence = self.root.log_probability
        self.revert(symbol_list_length)

        return math.exp(prob_sequence - prob_history)
    # end def

    def revert(self, symbol_count = 1):
        """ Restores the context tree to its state prior to a specified number of updates.
     
            - `num_symbols`: the number of updates (symbols) to revert. (Default of 1.)
        """

        # Traverse the tree from leaf to root according to the context.
        for i in xrange(0, symbol_count):
            # Check if we have updates to revert.
            if len(self.history) == 0:
                return
            # end if

            # Get the most recent symbol and delete from the history.
            symbol = self.history.pop()

            # Traverse the tree from leaf to root according to the context. Update the
            # probabilities and symbol counts for each node. Delete unnecessary nodes.
            if len(self.history) >= self.depth:
                self.update_context()

                # Step backwards through the nodes in the context in reverse context order.
                # (Only go as deep as the current tree depth, though.)
                for context_node in reversed(self.context[:self.depth]):
                    context_node.revert(symbol)
                # end for
            # end if
        # end for
    # end def

    def revert_history(self, symbol_count = 1):
        """ Shrinks the history without affecting the context tree.

            (Called `revertHistory` in the C++ version.)
        """

        assert symbol_count > 0, "The given symbol count should be greater than 0."
        history_length = len(self.history)
        assert history_length >= symbol_count, "The given symbol count must be greater than the history length."

        new_size = history_length - symbol_count
        self.history = self.history[:new_size]
    # end def

    def size(self):
        """ Returns the number of nodes in the context tree.
        """

        # Return the value stored and updated by the children nodes.
        return self.tree_size
    # end def

    def update(self, symbol_list):
        """ Updates the context tree with a new (binary) symbol, or a list of symbols.
            Recalculates the log weighted probabilities and log KT estimates for each affected node.

            - `symbol_list`: the symbol (or list of symbols) with which to update the tree.
                              (The context tree is updated with symbols in the order they appear in the list.)
        """

        # Ensure that we have a list, by making this a list if it's a single symbol.
        if type(symbol_list) != list:
            symbol_list = [symbol_list]
        # end if

        # Traverse the tree from leaf to root according to the context.
        for symbol in symbol_list:
            # Update the probabilities and symbol counts for each node.
            symbol = int(symbol)
            if len(self.history) >= self.depth:
                self.update_context()

                # Step backwards through the nodes in the context in reverse context order.
                # (Only go as deep as the current tree depth, though.)
                for context_node in reversed(self.context[:self.depth]):
                    context_node.update(symbol)
                # end for
            # end if

            # Add this symbol to the history.
            self.update_history(symbol)
        # end for
    # end def

    def update_context(self):
        """ Calculates which nodes in the context tree correspond to the current
            context, and adds them to `context` in order from root to leaf.

            In particular, `context[0]` will always correspond to the root node
            and `context[self.depth]` corresponds to the relevant leaf node.

            Creates the nodes if they do not exist.

            (Called `updateContext` in the C++ version.)
        """

        # Ensure that the length of the history is greater than or equal to the tree depth for safety.
        assert len(self.history) >= self.depth, "The history length must be greater than the tree depth."

        # Traverse the tree from root to leaf according to the context.
        # Save the path taken and create new nodes as necessary.
        self.context = [self.root]
        node = self.root
        update_depth = 1
        for symbol in reversed(self.history):
            # Find the relevant child node of the current node for the current symbol, if it exists.
            symbol = int(symbol)
            if symbol in node.children:
                node = node.children[symbol]
            else:
                # No child exists for this symbol.

                # Create a new node for the context, and add it into the tree under the current symbol.
                new_node = CTWContextTreeNode(tree = self)
                node.children[symbol] = new_node

                # Increase the size of the tree by 1, for the new node.
                self.tree_size += 1

                # Move onto this new node.
                node = new_node
            # end if

            # Add the node to the context path.
            self.context += [node]

            # Have we hit the end of the update depth yet?
            update_depth += 1
            if update_depth > self.depth:
                # Yes. Stop updating the context.
                break
            # end if
        # end for
    # end def

    def update_history(self, symbol_list):
        """ Appends a symbol (or a list of symbols) to the tree's history without updating the tree.

            - `symbol_list`: the symbol (or list of symbols) to add to the history.

            (Called `updateHistory` in the C++ version.)
        """

        # Ensure that we have a list, by making this a list if it's a single symbol.
        if type(symbol_list) != list:
            symbol_list = [symbol_list]
        # end if

        self.history += symbol_list
    # end def
# end class