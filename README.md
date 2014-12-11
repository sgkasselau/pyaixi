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

pypy aixi.py -v conf/rock_paper_scissors_fast.conf


Or if you have Jython (e.g. version 2.7) installed:

jython27 aixi.py -v conf/rock_paper_scissors_fast.conf


NOTE: it is highly recommended to use the PyPy http://pypy.org Python interpreter to
run code from this package, as this typically provides an order-of-magnitude run-time
improvement over using the standard CPython interpreter.

(This is unfortunately still an order of magnitude slower than the C++ version, though.
Jython is an order of magnitude slower again, but this code is functional under Jython,
if an integration with Java code is desired.)



This example will perform 500 interactions of the agent with the environment, with the agent
exploring the environment by trying permitted actions at random, and learning from
the related observations and rewards.

Then, the agent will use what it has learnt to maximise its reward in the following
500 interactions. (Exploration is typically quite quick, while using that gained knowledge
to choose the best action possible is typically much slower.)


For this particular environment, an average reward greater than 1 means the agent is winning
more than it is losing.

(A score ranging from 0.96 to 1.07 is typical for the MC-AIXI-CTW agent, depending on the random seed given.)


Further example environments can be found in the `environments` directory:

 - coin_flip            - A simulation of a biased coin flip
 - extended_tiger       - An extended version of the Tiger-or-Gold door choice problem.
 - kuhn_poker           - A simplified, zero-sum version of poker.
 - maze                 - A two-dimensional maze.
 - oscillator           - An oscillating machine that cycles between high and low states.
 - rock_paper_scissors  - Rock Paper Scissors.
 - tic_tac_toe          - Tic Tac Toe.
 - tiger                - A choice between two doors. One door hides gold; the other, a tiger.
 - two_coin_flip        - Simulates two biased coin flips. (Can the agent do better than random chance?)

Similarly-named environment configuration files for these environments can be found in the
`conf` directory, and run by replacing `rock_paper_scissors_fast.conf` in the commands
listed above with the name of the appropriate configuration file.


Script usage
------------

Usage: python aixi.py [-a | --agent <agent module name>]
                      [-c | --compare <agent to compare with e.g. aixi_uniform_random, for hypothesis testing>]
                      [-d | --explore-decay <exploration decay value, between 0 and 1>]
                      [-e | --environment <environment module name>]
                      [-h | --agent-horizon <search horizon>]
                      [-l | --learning-period <cycle count>]
                      [-m | --mc-simulations <number of simulations to run each step>]
                      [-n | --non-learning-only (whether to collate non-learning statistics only)]
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

The following agent class can be found in the `agent` directory:

 - aixi_uniform_random - a concrete/usable agent that just selects actions in a (uniformally) random way.
                         Useful for statistical hypothesis testing with the -c agent comparison option.

 - mc_aixi             - a base AIXI-style agent class that uses Monte Carlo (i.e. random sampling) search
                         over any prediction/learning model that implements the base pyaixi.Model interface.
                         (Needs to be subclassed and provided with a model before it can be used as an agent.)

 - mc_aixi_ctw         - a concrete/usable agent implementing the Monte Carlo-AIXI-Context Tree Weighting algorithm,
                         and the main agent/algorithm of interest in this project.
                         A useful reference for writing your own agents that inherit from the base
                         MC_AIXI_Agent/mc_aixi class.


The prediction/learning models available for agents to use can be found in the `prediction` directory:

 - ctw_context_tree - an implementation of Context Tree Weighting-style context trees.
                      This inherits from the Model base class, and is usable with agents
                      that inherit from the MC_AIXI class.


The particular search algorithm used by the MC_AIXI base class is found in the `search` directory:

 - monte_carlo_search_tree - an implementation of Monte Carlo search trees.


New agents need to inherit from the base `agent.Agent` class (or one of its subclasses e.g. agent.MC_AIXI_Agent),
and provide the methods listed within to interact with the currently-configured environment.

To use your own agent instead of the default `mc_aixi_ctw` agent in a configuration file,
use the `agent` key to specify the Python module name of your agent.

Alternatively, you can override the default/the configuration file value, by using
the '-a'/'--agent' option on the command line.


Comparing agent performance
---------------------------

To test a hypothesis about which of two agents performs best in a particular environment, agents can be compared
using the '-c/--compare' option, as for the following in the Oscillator environment:

pypy aixi.py -n -v conf/oscillator.conf -c aixi_uniform_random

This compares the default MC-AIXI-CTW agent (with results displayed as agent 'A') against another agent
named aixi_uniform_random (and displayed as agent 'B'), which is a simple agent that only chooses actions in a
(uniformally) random way.

As well as the usual running reward totals and average display, a variety of statistics about the rewards earned
will be displayed every 50 cycles during the non-learning period (or during the entire session by omitting
the '-n/--non-learning-only' option) so that you're not mislead about an agent's performance by a
particularly good or bad session.

In addition to count, minimum, maximum, median, average/mean, variance, skewness, and kurtosis values displayed
for the population of each agent's reward values, the following statistical tests are performed:

Mann-Whitney U-test:

    Tests whether the two populations are likely equal in distribution of numerical values (when close to 100%), or if
    one tends to have greater numerical values than the other (when close to 0%.)

    When close to 0%, read this as meaning that one agent is, overall, performing 'better' (with higher reward values)
    than the other.

    Use the displayed mean and median reward values of each agent to work out which is performing best
    (e.g. higher is typically better), and the following tests to see how this might be improved e.g. by working on
    minimising the variation between the minimum and maximum reward values, or by increasing one or both.

    http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test


Mood's median test:

    Tests whether the two populations have the same scale/range in values i.e. whether rewards are scattered between the
    same minimum and maximum values in each population.

    The higher the probability, the more likely the two populations are equal in scale/range. The lower the probability,
    the more likely that one agent is uniformally (best, worst, and average-case) performing better or worse than the other.

    Use the displayed minimum and maximum values of each agent to work out which likely has the greater scale/range in this way.

    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mood.html


Fligner's mean and median tests: 

    Tests the variance (distance from the mean or median) of values in the compared reward values of each
    agent, assuming (respectively) a symmetrical distribution of reward values, or no such distribution.

    (The difference between the two indicates roughly whether one agent's reward values are
     symmetrically distributed or not.)

    The higher the probability, the more likely the two populations are equal in variance. The lower the probability,
    the more likely that one agent is showing comparatively variable performance.
    (Use the recorded per-agent variance values displayed alongside these tests to determine which.)

    If comparing against a random agent (i.e. aixi_uniform_random), a low probability might indicate that the original
    agent is performing in a highly consistent manner i.e. consistently good or bad.

    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.fligner.html


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
Geoff Kassel, via:

geoffkassel_at_gmail_dot_com