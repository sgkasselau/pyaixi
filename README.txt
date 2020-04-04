pyaixi
======

Description
-----------

A pure Python implementation of the Monte Carlo-AIXI-Context Tree Weighting (MC-AIXI-CTW)
artificial intelligence algorithm.

This is an approximation of the AIXI universal artificial intelligence algorithm, which
describes a model-based, reinforcement-learning agent capable of general learning.


A more in-depth description of the MC-AIXI-CTW algorithm can be found here:

J.Veness, K.S.Ng, M.Hutter, W.Uther, D.Silver,
A Monte Carlo AIXI Approximation,
Journal of Artificial Intelligence Research, 40 (2011) 95-142
http://dx.doi.org/10.1613/jair.3125
Free TechReport version: http://arxiv.org/abs/0909.0801
BibTeX: http://www.hutter1.net/official/bib.htm#aixictwx


Motivation
----------

Providing a pure Python implementation of the MC-AIXI-CTW algorithm is intended to:

- help make the implementation of AIXI-approximate algorithms more accessible to people
  without a C++ background

- permit easier use of the MC-AIXI-CTW algorithm (and components) in other Python-based
  AI projects, and

- permit faster prototyping of new AIXI-approximate algorithms via Python's comparative
  linguistic simplicity.


Getting started
---------------

To try the example `Rock Paper Scissors` environment, run the following in the
base directory of this package.

From the Linux/Unix/Mac console:

python aixi.py -v conf/rock_paper_scissors_fast.conf


On Windows:

python aixi.py -v conf\rock_paper_scissors_fast.conf


Or if you have PyPy (e.g. version 1.9) installed on Linux/Unix/Mac:

pypy-c1.9 aixi.py -v conf/rock_paper_scissors_fast.conf


NOTE: it is highly recommended to use the PyPy http://pypy.org Python interpreter to
run code from this package, as this typically provides an order-of-magnitude run-time
improvement over using the standard CPython interpreter.

(This is unfortunately still an order of magnitude slower than the C++ version, though.)


This example will perform 500 interactions of the agent with the environment, with the agent
exploring the environment by trying permitted actions at random, and learning from
the related observations and rewards.

Then, the agent will use what it has learnt to maximise its reward in the following
500 interactions. (Exploration is typically quite quick, while using that gained knowledge
to choose the best action possible is typically much slower.)


For this particular environment, an average reward greater than 1 means the agent is winning
more than it is losing.

(A score ranging from 1.02 to 1.04 is typical, depending on the random seed given.)


Further example environments can be found in the `environments` directory:

 - coin_flip            - A simulation of a biased coin flip
 - extended_tiger       - An extended version of the Tiger-or-Gold door choice problem.
 - kuhn_poker           - A simplified, zero-sum version of poker.
 - maze                 - A two-dimensional maze.
 - rock_paper_scissors  - Rock Paper Scissors.
 - tic_tac_toe          - Tic Tac Toe
 - tiger                - A choice between two doors. One door hides gold; the other, a tiger.

Similarly-named environment configuration files for these environments can be found in the
`conf` directory, and run by replacing `rock_paper_scissors_fast.conf` in the commands
listed above with the name of the appropriate configuration file.


Script usage
------------

Usage: python aixi.py [-a | --agent <agent module name>]
                      [-d | --explore-decay <exploration decay value, between 0 and 1>]
                      [-e | --environment <environment module name>]
                      [-h | --agent-horizon <search horizon>]
                      [-l | --learning-period <cycle count>]
                      [-m | --mc-simulations <number of simulations to run each step>]
                      [-o | --option <extra option name>=<value>]
                      [-p | --profile]
                      [-r | --terminate-age <number of cycles before stopping the run>]
                      [-t | --ct-depth <maximum depth of predicting context tree>]
                      [-x | --exploration <exploration factor, greater than 0>]
                      [-v | --verbose]
                      [<environment configuration file name to load>]


Adding new environments
-----------------------

The environments in the `environments` directory all inherit from
a base class, `environment.Environment`, found in the base package directory.

New environments will need to inherit this class, and provide the methods
of this class (as well as any internal logic) to interact with the agent.

You'll also need to construct a new configuration file for this environment,
making sure to give the name of your new environment in the `environment` key.


Adding new agents
-----------------

The only (for now) provided agent class can be found in the `agent` directory:

 - mc_aixi_ctw - an agent implementing the Monte Carlo-AIXI-Context Tree Weighting algorithm.


The prediction algorithm used by this agent can be found in the `prediction` directory:

 - ctw_context_tree - an implementation of Context Tree Weighting context trees.


The search algorithm used is found in the `search` directory:

 - monte_carlo_search_tree - an implementation of Monte Carlo search trees.


New agents need to inherit from the base `agent.Agent` class, and provide the methods
listed within to interact with the currently-configured environment.

To use your own agent instead of the default `mc_aixi_ctw` agent in a configuration file,
use the `agent` key to specify the Python module name of your agent.

Alternatively, you can override the default/the configuration file value, by using
the '-a'/'--agent' option on the command line.


Similar projects
----------------

This package is based on the C++ implementation of the MC-AIXI-CTW algorithm seen here:

https://github.com/moridinamael/mc-aixi


Another implementation of MC-AIXI-CTW can be found here:

Joel Veness's personal page: http://jveness.info/software/default.html


License
-------

Creative Commons Attribution ShareAlike 3.0 Unported. (CC BY-SA 3.0)

Please see `LICENSE.txt` for details.

If permitted in your legal domain (as this package is arguably a substantive
derivative of another CC BY-SA 3.0 package, hence the licensing terms above,
and the legal compatibility of CC BY-SA 3.0 with other open-source licences is currently
unknown), the author of this package permits alternate licensing under your
choice of either the LGPL 3.0 or the GPL 3.0.


Contact the author
------------------

For further assistance or to offer constructive feedback, please contact the author,
SG Kassel, via:

sg_dot_kassel_dot_au_at_gmail_dot_com