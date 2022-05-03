# Change log

## 1.0.4

- Added support for Python 3.x
- Fixed an issue with how the random seed was set when no seed value is supplied

## 1.0.3

- Rearranged the package's directory hierarchy and module imports to fix import issues with the `aixi.py` script when run from the final installed location.

## 1.0.2

- Improved messaging and documentation about long pauses during (non-random-exploration phase) action selection.

## 1.0.1

- Fixed imports so that this package works out of the box on all platforms without installation.

## 1.0.0

- Initial import of Python code based on the C++ MC-AIXI-CTW implementation at https://github.com/moridinamael/mc-aixi making the following changes in translation:

  - `main.cpp/aixi` executable to `aixi.py`:

    - The Python script equivalent takes optional command-line configuration options that override the
      option value with the same name from the given configuration file.

    - Added agent specification into the configuration options, for specifying the use of an
      agent based on a (new) base Agent class.

    - Any class can be used as an environment, so long as it inherits from the base Environment
      class, and if it's located within the Python system/package search path.

    - Added a profiling option ('-p'/'--profile') to generate execution-time statistics useful
      in finding which parts of the algorithm are consuming the most time, for subsequent
      optimisation work.

  - `agent.cpp` to `agent.py` and `agents/mc_aixi_ctw.py`:

    - split into two parts: a base `Agent` class in `agent.py`, and the parts more specific
      to the MC-AIXI-CTW algorithm in `agents/mc_aixi_ctw.py`.

    - the base `Agent` class is intended for agent classes to use to inherit (and override) basic
      environment interoperability methods.

    - accessor methods replaced with direct property access for simplicity.
      (If needed later, these properties can be transparently turned into calls to accessors.)

  - `environment.cpp` to `environment.py`:
  - `search.cpp` to `search/monte_carlo_search_tree.py`:

    - accessor methods replaced with direct property access for simplicity.

  - `predict.cpp` to `prediction/ctw_context_tree.py`:

    - tweaked the algorithm to cache the size of the context tree in the top-level tree object,
      as this provides a significant time performance improvement during exploration, with
      no observed performance decrease in other circumstances.

    - made smaller, non-algorithmic tweaks to improve time performance in this critical path
      code.

    - accessor methods replaced with direct property access for simplicity.

  - `util.cpp` to `util.py`:

    - translated only functions that didn't already have a Python equivalent.

    - added a enumeration generating function to replicate C++ enumeration types, while
      also providing membership checks and iteration over the range of defined values.