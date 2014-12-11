#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines a class for an AIXI agent that just makes (uniformly) random actions from the range of permitted actions.
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

from pyaixi import agent

class AIXI_Uniform_Random_Agent(agent.Agent):
    """ This class represents an AIXI agent that makes (uniformly) random actions.
    """

    # Instance methods.

    def search(self):
        """ Returns a random action uniformly from the range of permitted actions.
            (`generate_random_action` generates random actions uniformly.)
        """

        return self.generate_random_action()
    # end def
# end class