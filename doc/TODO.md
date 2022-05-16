# TODO

- [x] Remove the support for Python 2, as it's dead now
  - [x] Update the source files to only support Python 3 (specifically, Python 3.9 for now)
  - [x] Remove `six.py`
  - [x] Change `setup.py` accordingly 
  - [x] Update the `README.md` to indicate that this project only supports Python 3.9 or above
- [x] Update the version of this package to 2.0.0 in `setup.py`, because the removal of the support for python 2 is a major change that can cause incompatibility issues

## Refactoring

- [x] To improve readability, convert `.txt` files to `.md` files
- [x] Refactor code to be compliant with pep8
  - I also used [black](https://github.com/psf/black) to do this
  - I didn't change `six.py`, as this is an external module that will not be needed once Python 2 support is removed
- [x] Change `setup.py` to use more appropriate keywords, like "universal intelligence" and "AGI"
- [ ] Move `agent.py` and `environment.py` inside their respective packages
- [ ] Rename this project to `aixipy`
- [ ] Rename the methods (e.g. `perform_action` to `step`) to be more consistent with Gym and other popular Python RL libraries.

## Issues

- [x] Make the agent and environment classes actually abstract
- [ ] All instances variables should be initialized inside `__init__`
- [ ] Don't use custom enumerations if they are unnecessary
- [ ] Do we really need the lines `PROJECT_ROOT = os.path.realpath(os.path.join(os.pardir, os.pardir)); sys.path.insert(0, PROJECT_ROOT)` in each module? See https://github.com/sgkasselau/pyaixi/issues/1.
- [ ] Fix all TODOs in the modules
- [ ] `__init__.py` of the `pyaixi` package exports `"aixi"` but it's not defined there.
- [ ] In the function `generate_random_symbols_and_update` of `ctw_context_tree.py`, convert `range(0, symbol_count)` to `range(symbol_count)`


## Bugs

- [x] Fix bug in `ctw_context_tree.py`, function `generate_random_symbols`
  - It seems that this method is never called in this project, that's why this bug was probably never found
  - Fixed by not wrongly passing `symbol_list` to `generate_random_symbols_and_update`, which expects only the `symbol_count` - the C++ version passes a reference to the vector of symbols `symbols` (which in the Python version is called `symbol_list`) to the method `genRandomSymbolsAndUpdate`, which then resizes this vector and resets its values probabilistically - in Python, we don't need to pass `symbol_list` to `generate_random_symbols_and_update` (although we could and then change its elements), but we can simply create a new list with the new values and return it, which is what was already being done.
- [ ] Fix IndexError bugs that appear (AT LEAST) in the module `pyaixi/environment.py`. See function `minimum_reward`.

## Features

- [ ] Implement tests and test against different Python versions (e.g. with tox)
- [ ] Use type hints
- [ ] Add `doc/LITERATURE.md` all the relevant papers
- [ ] The results of experiments could be dumped to a file or DB for reproducibility
- [ ] Add visualization functionality
- [ ] Provide benchmarks that compare this implementation with the C++, JS, Java, and Go ones, and with PyPy
- [ ] Could the environments be represented with gym?
- [ ] Implement the typical environments, like CartPole (is that possible?)
- [ ] Pacman environment (which is implemented in the C++ version [here](https://github.com/moridinamael/mc-aixi/blob/master/conf/pacman.conf))
