#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines an environment for a two-dimensional maze.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import sys

from pyaixi import environment, util

# Define a enumeration to represent agent interactions with the environment,
# such as going left, up, right, or down.
maze_action_enum = util.enum('aLeft', 'aUp', 'aRight', 'aDown')

# Define a enumeration to represent environment observations: either a cell
# is empty, or has a wall in various (bit) positions.
maze_observation_enum = util.enum(oNull = 0, oLeftWall = 1, oUpWall = 2, oRightWall = 4, oDownWall = 8)

# Define an enumber to represent observation encoding constants.
maze_observation_encoding_enum = util.enum('cUninformative', 'cWalls', 'cCoordinates');

# Define some shorthand notation for ease of reference.
aLeft          = maze_action_enum.aLeft
aUp            = maze_action_enum.aUp
aRight         = maze_action_enum.aRight
aDown          = maze_action_enum.aDown

oNull          = maze_observation_enum.oNull
oLeftWall      = maze_observation_enum.oLeftWall
oUpWall        = maze_observation_enum.oUpWall
oRightWall     = maze_observation_enum.oRightWall
oDownWall      = maze_observation_enum.oDownWall

cUninformative = maze_observation_encoding_enum.cUninformative
cWalls         = maze_observation_encoding_enum.cWalls
cCoordinates   = maze_observation_encoding_enum.cCoordinates

# Define some maze layout constants.
cWall         = '@'
cTeleportTo   = '*'
cTeleportFrom = '!'
cEmpty        = '&'


class Maze(environment.Environment):
    """ A two-dimensional maze environment.

        The agent is able to move through the maze in each of the four cardinal directions
        (`aLeft`, `aUp`, `aRight`, `aDown`). The user is able to specify (via a
        configuration file) the dimensions, layout, and rewards of the maze as well
        as the type of observations given to the agent. In particular, the maze
        is a certain number of rows high and columns wide, each cell in the maze is
        of a certain type and has a certain reward.

        The type of each cell determines what happens to the agent when it attempts to move
        into the cell:

        - Wall (`cWall`): represents an impassable cell, attempting to move into
          a wall will result in the agent remaining at its current position.

        - Empty (`cEmpty`): an empty cell through which the agent can freely
          pass.

        - Teleport from (`cTeleportFrom`): represents a cell which, upon entry,
          will randomly teleport the agent to another cell of type `cTeleportTo`.

        - Teleport to (`cTeleportTo`): A cell which can be teleported to.
          Otherwise, the cell acts identically to an empty cell. The maze MUST
          contain at least one of these cells.

        The reward for each cell describes the reward received by the agent when it
        attempts to move into that cell. For example, when the agent attempts to move
        into a wall square, it gets the reward from the wall cell despite the fact
        that it does not end its turn in that cell. Similarly, when the agent moves
        into a `cTeleportFrom` cell it gets the reward from that cell rather
        than from the cell it will be teleported to.

        Finally, the user can choose one of several observation encodings to give
        to the agent:

        - Uninformative (`cUninformative`): The observation is a single unchanging value.

        - Walls (`cWalls`): The observation encodes the presence of walls in
          adjacent squares. This is encoded as a sum of the flags: `oDownWall`,
          `oLeftWall`, `oRightWall`, `oUpWall`, each of which indicates the presence
          of a wall in the corresponding direction.

        - Coordinates (`cCoordinates`): The observation gives the coordinates
          of the cell occupied by the agent. This is encoded as row * num_cols + col.

        Domain characteristic:

        - environment: "maze"

        - maximum action: 3 (2 bits)

        - maximum observation:

          - Uninformative: 0 (1 bit)

          - Walls: 15 (4 bits)

          - Coordinates: self.num_rows * self.num_cols - 1

        - maximum reward: maximum value in maze rewards (`self.max_reward`)

        Configuration options:

        - maze-num-rows: the number of rows in the maze.

        - maze-num-cols: the number of columns in the maze.

        - maze-rewards#: comma-separated list of rewards for each square in row #. If
          the agent enters (or attempts to enter) a particular square it receives
          the corresponding reward. 1 <= # <= maze-num-rows.

        - maze-layout#: The layout of row # of the maze (1 <= # <= maze-num-rows).

          Contains maze-num-cols symbols as follows:
          - `cWall`: indicates the square cannot be entered (a.k.a. "a wall").
          - `cTeleportTo`: indicates the square can be entered and teleported to.
          - `cEmpty`: indicates the square can be entered but not teleported to.
          - `cTeleportFrom`: indicates the square can be entered, and that doing
            so will randomly teleport the agent to a `cTeleportTo` square before
            the next turn (the agent receives the reward before the teleportation
            occurs).

        - maze-observation-encoding: Specifies the type of observations the agent receives
          - uninformative: the agent receives the same observation each cycle.
          - walls: the agent receives an observation specifying whether there are
                   walls "@" above, below, left, or right of its current position.
          - coordinates: the observation specifies the coordinates of the agent in
                         the maze.
    """

    # Instance methods.

    def __init__(self, options = {}):
        """ Construct the Maze environment from the given options.

             - `options` is a dictionary of named options and their values.
        """

        # Set up the base environment.
        environment.Environment.__init__(self, options = options)

        # Configure the environment based on the options.
        self.configure(options)

        # Define the acceptable action values.
        self.valid_actions = maze_action_enum.keys()

        # Define the acceptable observation values.
        self.valid_observations = xrange(0, self.max_observation() + 1)

        # Define the acceptable reward values.
        self.valid_rewards = xrange(0, self.max_reward + 1)

        # Assign the location of the agent randomly.
        self.teleport_agent()

        # Set the initial reward and percept.
        self.reward = 0
        self.calculate_observation()
    # end def

    def calculate_observation(self):
        """ Determines the observation to give to the agent based on its current
            location (`self.row` and `self_col`) and the observation encoding
            (`self.observation_encoding`). Store the observation in `self.observation`.

            (Called `calculate_observation` in C++ version.)
        """

        if self.observation_encoding == cUninformative:
            # Uninformative observation: agent always receives same observation.
            self.observation = oNull
        elif self.observation_encoding == cWalls:
            # Agent observes adjacent walls.
            self.observation = 0
            if self.col == 0 or self.maze_layout[self.row][self.col - 1] == cWall:
                self.observation += oLeftWall
            # end if
            if self.row == 0 or self.maze_layout[self.row - 1][self.col] == cWall:
                self.observation += oUpWall
            # end if
            if self.col + 1 == self.num_cols or self.maze_layout[self.row][self.col + 1] == cWall:
                self.observation += oRightWall
            # end if
            if self.row + 1 == self.num_rows or self.maze_layout[self.row + 1][self.col] == cWall:
                self.observation += oDownWall
            # end if
        elif self.observation_encoding == cCoordinates:
            # Agent observes the coordinates of its current square.
            self.observation = self.row * self.num_cols + self.col
        # end if
    # end def

    def configure(self, options):
        """ Configures the maze based on the given configuration options.
            May exit if the configuration is not validly formatted.
        """

        # Get the dimensions of the maze from the options.
        self.num_rows = options.get("maze-num-rows", None)
        if self.num_rows is None:
            sys.stderr.write("ERROR: configuration does not contain a 'maze-num-rows' value. Exiting." + \
                             os.linesep)
            sys.exit(1)
        # end if
        self.num_rows = int(self.num_rows)

        self.num_cols = options.get("maze-num-cols", None)
        if self.num_cols is None:
            sys.stderr.write("ERROR: configuration does not contain a 'maze-num-cols' value. Exiting." + \
                             os.linesep)
            sys.exit(1)
        # end if
        self.num_cols = int(self.num_cols)

        assert(self.num_rows > 0)
        assert(self.num_cols > 0)

        # Get the type of observations to give.
        encoding = options.get("maze-observation-encoding", "uninformative")
        if encoding == "uninformative":
            self.observation_encoding = cUninformative
        elif encoding == "walls":
            self.observation_encoding = cWalls
        elif encoding == "coordinates":
            self.observation_encoding = cCoordinates
        else:
            # We've got an unknown observation.
            sys.stderr.write("ERROR: Unknown observation encoding: '%s'" % str(encoding) + os.linesep)
            sys.exit(1)
        # end if

        # Variable indicating whether there are any squares which can be teleported to.
        teleport_impossible = True

        # Used to record the minimum and maximum rewards in the maze.
        min_reward = float('inf')
        self.max_reward = float('-inf')

        # Allocate and parse maze, determine if teleportation is possible and the
        # range of rewards.
        self.maze_rewards = {}
        self.maze_layout = {}
        self.teleport_to_locations = []
        for r in xrange(0, self.num_rows):
            # Get the reward string for the current row from the options.
            reward_option_name = "maze-rewards%d" % (r + 1)
            rewards = options.get(reward_option_name, None)

            if rewards is None:
                sys.stderr.write("ERROR: configuration does not contain a " + \
                                 "'%s' value as a num_rows value of '%d' implies. Exiting." % \
                                 (reward_option_name, self.num_rows) + os.linesep)
                sys.exit(1)
            # end if

            # Get the layout string for the current row from the options.
            layout_option_name = "maze-layout%d" % (r + 1)
            layout = options.get(layout_option_name, None)

            if layout is None:
                sys.stderr.write("ERROR: configuration does not contain a " + \
                                 "'%s' value as a num_rows value of '%d' implies. Exiting." % \
                                 (layout_option_name, self.num_rows) + os.linesep)
                sys.exit(1)
            # end if

            # Parse the reward and layout strings, and record appropriate properties of the maze.

            # Ensure the maze layout and rewards have a dictionary in each row to hold the
            # column values.
            if r not in self.maze_layout:
                self.maze_layout[r] = {}
            # end if

            if r not in self.maze_rewards:
                self.maze_rewards[r] = {}
            # end if

            # Split the layout string into a list, and split the reward strings into a list, splitting on commas.
            layout_list = list(layout)
            rewards_list = rewards.split(',')

            # If we don't have exactly enough entries, report this to the user.
            if len(layout_list) != self.num_cols:
                sys.stderr.write("ERROR: configuration value '%s' (%s)" % (layout_option_name, layout) + \
                                 "contains too %s entries. (Needs '%d'.) Exiting." % \
                                 ("few" if len(layout_list) < self.num_cols else "many", self.num_cols) + \
                                 os.linesep)
                sys.exit(1)
            # end if

            if len(rewards_list) != self.num_cols:
                sys.stderr.write("ERROR: configuration value '%s' (%s)" % (reward_option_name, rewards) + \
                                 "contains too %s entries. (Needs '%d'.) Exiting." % \
                                 ("few" if len(rewards_list) < self.num_cols else "many", self.num_cols) + \
                                 os.linesep)
                sys.exit(1)
            # end if

            # Turn the layout and reward strings into dictionary/two-dimensional array entries,
            # checking each value as it's inspected.
            for c in xrange(0, self.num_cols):
                #
                this_layout = layout_list[c]
                this_reward = int(rewards_list[c])

                self.maze_layout[r][c]  = this_layout
                self.maze_rewards[r][c] = this_reward

                # Is this the teleport-to character?
                if this_layout == cTeleportTo:
                    # Yes. Teleporting is possible in this maze.
                    teleport_impossible = False

                    # Record this location in the teleport-to location list.
                    self.teleport_to_locations += [(r, c)]
                # end if

                # Update the minimum and maximum reward if required.
                min_reward = min_reward if min_reward < this_reward else this_reward
                self.max_reward = self.max_reward if self.max_reward > this_reward else this_reward
            # end for
        # end for

        # Exit with an error message if it is impossible for the agent to teleport anywhere.
        if teleport_impossible:
            sys.stderr.write("ERROR: There must be at least one square the agent can teleport to.")
            sys.exit(1)
        # end if

        # Adjust rewards so they begin at 0.
        self.max_reward -= min_reward
        for r in xrange(0,  self.num_rows):
            for c in xrange(0, self.num_cols):
                self.maze_rewards[r][c] = self.maze_rewards[r][c] - min_reward
            # end for
        # end for
    # end def

    def max_observation(self):
        """ Returns the maximum observation that can be given to the agent.
            Depends on the observation encoding (`self.observation_encoding`) and
            (potentially) the dimensions of the maze (`self.num_rows` and `self.num_cols`.)

            (Called `max_observation` in the C++ version.)
        """
        if self.observation_encoding == cUninformative:
            # Only one observation
            return oNull
        elif self.observation_encoding == cWalls:
            # Maximum observation is walls on all sides
            return oLeftWall + oUpWall + oRightWall + oDownWall
        elif self.observation_encoding == cCoordinates:
            # Maximum observation is square at intersection of last row and column
            return (self.num_rows * self.num_cols) - 1
        # end if
    # end def

    def perform_action(self, action):
        """ Receives the agent's action and calculates the new environment percept.

            (Called `performAction` in the C++ version.)
        """

        assert self.is_valid_action(action)

        # Save the action.
        self.action = action

        # Set the flags for this action.
        self.teleported = False
        self.wall_collision = False

        # Calculate the square the agent is attempting to move to, making sure they
        # don't move outside the maze.
        self.row_to = (-1 if action == aUp else 0) + (1 if action == aDown else 0)
        self.row_to = min(max(self.row_to + self.row, 0), self.num_rows - 1)
        self.col_to = (-1 if action == aLeft else 0) + (1 if action == aRight else 0)
        self.col_to = min(max(self.col_to + self.col, 0), self.num_cols - 1)

        # Move the agent, making sure they don't walk into a wall.
        self.wall_collision = (self.maze_layout[self.row_to][self.col_to] == cWall)
        if not self.wall_collision:
            self.row = self.row_to
            self.col = self.col_to
        # end if

        # Teleport if appropriate.
        if self.maze_layout[self.row][self.col] == cTeleportFrom:
            self.teleport_agent()
        # end if

        # Calculate the reward for the square the agent *attempted* to move into,
        # regardless of whether they were able to move into it.
        self.reward = self.maze_rewards[self.row_to][self.col_to]

        # Calculate the observation for the square the agent is now in.
        # That is, after any movement or teleportation has occurred.
        self.calculate_observation()

        return (self.observation, self.reward)
    # end def

    def print(self):
        """ Returns a string indicating the current state of the environment, including
            the current location, observation, reward, and maze layout.
        """

        message = "row = %d" % self.row + ", col = %d" % self.col + \
                  ", observation = %d" % self.observation + \
                  ", reward = %d" % self.reward + \
                  (", teleported" if self.teleported else "") + \
                  (", wall collision" if self.wall_collision else "") + os.linesep

        for r in xrange(0, self.num_rows):
            for c in xrange(0, self.num_cols):
                if self.row == r and self.col == c:
                    message += "A"
                else:
                    message += self.maze_layout[r][c]
                # end if
            # end for
            message += os.linesep
        # end for

        return message
    # end def

    def teleport_agent(self):
        """ Randomly places the agent at any `cTeleportTo` location.

           (Called `teleportAgent` in the C++ version.)
        """

        self.teleported = True

        # This is altered from the C++ version to be far more efficient.
        # (e.g. instead of random search of the maze, use a random choice
        #  over a pre-computed list of possible destinations.)
        self.row, self.col = random.choice(self.teleport_to_locations)
    # end def
# end class