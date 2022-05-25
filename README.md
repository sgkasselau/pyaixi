# pyaixi


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  [![CC BY-SA 3.0][cc-by-sa-shield]][cc-by-sa]

## Description

A Python implementation of the _Monte Carlo-AIXI-Context Tree Weighting (MC-AIXI-CTW)_ artificial intelligence algorithm. 

This implementation, initially developed by [SG Kassel](https://github.com/sgkasselau) but now also developed by [nbro](https://github.com/nbro), is highly based on the C++ implementation (which you can find at the Marcus Hutter's site [here](http://www.hutter1.net/publ/aixictwxcode.zip)) by Daniel Visentin (and I think other Hutter's students).

MC-AIXI-CTW is an approximation of the AIXI universal artificial intelligence algorithm (proposed by [Marcus Hutter](http://www.hutter1.net/)), which  describes a model-based, reinforcement-learning agent capable of general learning.

A more in-depth description of the MC-AIXI-CTW algorithm can be found in the paper [A Monte Carlo AIXI Approximation](http://arxiv.org/abs/0909.0801) by  J. Veness et al., published in the [Journal of Artificial Intelligence Research](https://www.jair.org/index.php/jair), 40 (2011) 95-142.

## Motivation

Providing a pure Python implementation of the MC-AIXI-CTW algorithm is intended to:

- help make the implementation of AIXI-approximate algorithms more accessible to people without a C++ background

- permit easier use of the MC-AIXI-CTW algorithm (and components) in other Python-based AI projects, and

- permit faster prototyping of new AIXI-approximate algorithms via Python's comparative linguistic simplicity.

## Installation

1. Create a virtual environment (optional but recommended)

2. Install this package in editable mode: `pip install -e .`

## Getting started

To try the example "Rock Paper Scissors" environment, run the following in the base directory of this package.

From the Linux/Unix/Mac console:

    python aixi.py -v conf/rock_paper_scissors_fast.conf

On Windows:

    python aixi.py -v conf\rock_paper_scissors_fast.conf

Or if you have PyPy (e.g. version 1.9) installed on Linux/Unix/Mac:

    pypy-c1.9 aixi.py -v conf/rock_paper_scissors_fast.conf

**NOTE**: it is highly recommended using the [PyPy Python interpreter](http://pypy.org/) to run code from this package, as this typically provides an order-of-magnitude run-time improvement over using the standard CPython interpreter; this is unfortunately still an order of magnitude slower than the C++ version, though.

This example will perform 500 interactions of the agent with the environment, with the agent exploring the environment by trying permitted actions at random, and learning from the related observations and rewards.

Then, the agent will use what it has learnt to maximise its reward in the following 500 interactions. 

Exploration is typically quite quick, while using that gained knowledge to choose the best action possible is typically much slower.

For this particular environment, an average reward greater than 1 means the agent is winning more than it is losing.

(A score ranging from 1.02 to 1.04 is typical, depending on the random seed given.)

## Environments

More example environments can be found in the [`environments`](./pyaixi/environments) directory:

 - `coin_flip`            - A simulation of a biased coin flip
 - `extended_tiger`       - An extended version of the Tiger-or-Gold door choice problem.
 - `kuhn_poker`           - A simplified, zero-sum version of poker.
 - `maze`                 - A two-dimensional maze.
 - `rock_paper_scissors`  - Rock Paper Scissors.
 - `tic_tac_toe`          - Tic Tac Toe
 - `tiger`                - A choice between two doors. One door hides gold; the other, a tiger.

Similarly-named environment configuration files for these environments can be found in the `conf` directory, and run by replacing `rock_paper_scissors_fast.conf` in the commands listed above with the name of the appropriate configuration file.


## Script usage

Please, execute `python aixi.py --help` to see all the options or look inside [`aixi.py`](./aixi.py).

## Adding new environments

The environments in the `environments` directory all inherit from a base class, `environment.Environment`, found in the base package directory.

New environments will need to inherit this class, and provide the methods of this class (as well as any internal logic) to interact with the agent.

You'll also need to construct a new configuration file for this environment, making sure to give the name of your new environment in the `environment` key.


## Adding new agents

The only (for now) provided agent class can be found in the `agent` directory:

 - `mc_aixi_ctw` - an agent implementing the Monte Carlo-AIXI-Context Tree Weighting algorithm.
 
The prediction algorithm used by this agent can be found in the `prediction` directory:

 - `ctw_context_tree` - an implementation of Context Tree Weighting context trees.
 
The search algorithm used is found in the `search` directory:

 - `monte_carlo_search_tree` - an implementation of Monte Carlo search trees.
 
New agents need to inherit from the base `agent.Agent` class, and provide the methods  listed within to interact with the currently-configured environment.

To use your own agent instead of the default `mc_aixi_ctw` agent in a configuration file, use the `agent` key to specify the Python module name of your agent.

Alternatively, you can override the default configuration file value, by using  the `-a` (or `--agent`) option on the command line.

## Contributors

See [`doc/CONTRIBUTORS.md`](./doc/CONTRIBUTORS.md).

## License

See [`doc/LICENSE.md`](./doc/LICENSE.md).

## Similar projects

See [`doc/SIMILAR_PROJECTS.md`](./doc/SIMILAR_PROJECTS.md).

## Literature

See [`doc/LITERATURE.md`](./doc/LITERATURE.md).

[cc-by-sa]: https://creativecommons.org/licenses/by-sa/3.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/3.0/eg/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/CC%20BY--SA%203.0-CC%20BY--SA%203.0-red
