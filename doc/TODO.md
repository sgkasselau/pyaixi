# TODO

- [x] To improve readability, convert `.txt` files to `.md` files
- [ ] Refactor code to be compliant with pep8
- [ ] Remove the support for Python 2, as it's dead now
  - [ ] Change setup.py, remove six and update the source files to only support Python 3
- [ ] Fix all TODOs in the modules
- [ ] Fix bug in `ctw_context_tree.py`, function `generate_random_symbols`
- [ ] Fix IndexError bugs that appear (AT LEAST) in the module `pyaixi/agent.py`. See function `minimum_reward`.
- [ ] All instances variables should be initialized inside `__init__`
- [ ] Implement tests and test against different Python versions (e.g. with tox)
- [ ] Make the agent and environment classes actually abstract and move them inside their respective packages
- [ ] Add to the README all the relevant papers

## Features

- [ ] The results of experiments could be dumped to a file or DB for reproducibility
- [ ] Add visualization functionality
- [ ] Provide benchmarks that compare this implementation with the C++, JS, and Java ones
- [ ] Could the environments be represented with gym?
- [ ] Implement the typical environments, like CartPole (is that possible?)
