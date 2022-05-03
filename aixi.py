#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script for running AIXI-based agents in an environment, as configured by
given options or the given configuration file.

Usage: python aixi.py [-a | --agent <agent module name>]
                      [-d | --explore-decay <exploration decay value, between
                      0 and 1>]
                      [-e | --environment <environment module name>]
                      [-h | --agent-horizon <search horizon>]
                      [-l | --learning-period <cycle count>]
                      [-m | --mc-simulations <number of simulations to run
                      each step>]
                      [-o | --option <extra option name>=<value>]
                      [-p | --profile]
                      [-r | --terminate-age <number of cycles before stopping
                      the run>]
                      [-t | --ct-depth <maximum depth of predicting context
                      tree>]
                      [-x | --exploration <exploration factor, greater than 0>]
                      [-v | --verbose]
                      [<environment configuration file name to load>]
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import six.moves.configparser as configparser

# TODO: do not use just bare except
try:
    import cProfile as profile
except:
    import profile

try:
    import cStringIO as StringIO
except:
    try:
        import StringIO
    except:
        import io as StringIO

import datetime
import getopt
import inspect
import random
import sys

# Insert the current directory into the system search path, so that this
# package can be imported when this script is run directly from a release
# archive.

PROJECT_ROOT = os.path.realpath(os.curdir)
sys.path.insert(0, PROJECT_ROOT)


def interaction_loop(agent=None, environment=None, options={}):
    """The main agent/environment interaction loop.

    Each interaction cycle begins with the agent receiving an observation and
    reward from the environment.

    Subsequently, the agent selects an action and informs the environment.

    The interactions that took place are logged to the logger.
    When the cycle equals a power of two, a summary of the interactions is printed to
    the standard output.

    - `agent`: the agent object.
    - `environment`: the environment object.
    - `options`: the configuration options.

    (Called `mainLoop` in the C++ version.)"""

    # Apply a random seed (Default: 0)
    random.seed(int(options.get("random-seed", 0)))

    # Verbose output (Default: False)
    verbose = bool(options.get("verbose", False))

    # Determine exploration options. (Default: don't explore, don't decay.)
    explore_rate = float(options.get("exploration", 0.0))
    explore = explore_rate > 0
    explore_decay = float(options.get("explore-decay", 1.0))

    assert 0.0 <= explore_rate
    assert 0.0 <= explore_decay <= 1.0

    # Determine termination age. (Default: don't terminate)
    terminate_age = int(options.get("terminate-age", 0))
    terminate_check = terminate_age > 0
    assert 0 <= terminate_age

    # Determine the cycle after which the agent stops learning (if ever.)
    learning_period = int(options.get("learning-period", 0))
    assert 0 <= learning_period

    # Agent/environment interaction loop.
    cycle = 1
    while not environment.is_finished:
        # Check for agent termination.
        if terminate_check and agent.age > terminate_age:
            break

        # Save the current time to compute how long this cycle took.
        cycle_start = datetime.datetime.now()

        # Get a percept from the environment.
        observation = environment.observation
        reward = environment.reward

        # If we're outside the learning period, stop exploring.
        if 0 < learning_period < cycle:
            explore = False

        # Update the agent's environment model with the new percept.
        agent.model_update_percept(observation, reward)

        # Determine best exploitive action, or explore.
        explored = False
        if explore and (random.random() < explore_rate):
            # Yes, we're still exploring.
            # Generate a random action to explore.
            explored = True
            if verbose:
                # Tell the user the agent is exploring at random.
                print("Agent is trying an action at random...")
            action = agent.generate_random_action()
        else:
            # No, we're not still exploring.
            # Exploit our past learning to work out the best action.
            if verbose:
                # Tell the user we're not exploring, we're trying to choose
                # the best action.
                print(
                    "Agent is trying to choose the best action, which may "
                    "take some time..."
                )
            action = agent.search()

        # Send the action to the environment.
        environment.perform_action(action)

        # Update the agent's environment model with the chosen action.
        agent.model_update_action(action)

        # Calculate how long this cycle took.
        time_taken = datetime.datetime.now() - cycle_start

        # Log this cycle.
        message = "%d, %s, %s, %s, %s, %f, %d, %f, %s, %d" % (
            cycle,
            str(observation),
            str(reward),
            str(action),
            str(explored),
            explore_rate,
            agent.total_reward,
            agent.average_reward(),
            str(time_taken),
            agent.model_size(),
        )

        print(message)

        # Print to standard output when cycle == 2^n or on verbose option.
        if verbose or (cycle & (cycle - 1)) == 0:
            message = (
                "cycle: %s" % str(cycle)
                + os.linesep
                + "average reward: %f" % agent.average_reward()
            )
            if explore:
                message += (
                    os.linesep
                    + "explore rate: %f" % float(explore_rate)
                    + os.linesep
                )

            print(message)

        # Print environment state if verbose option is true.
        if verbose:
            print(environment.print())

        # Update exploration rate.
        if explore:
            explore_rate *= explore_decay

        # Update the cycle count.
        cycle += 1

    # Print summary to standard output.
    message = (
        "SUMMARY:"
        + os.linesep
        + "agent age: %d" % agent.age
        + os.linesep
        + "average reward: %f" % agent.average_reward()
    )

    print(message)


def main(argv):
    """Entry point of the program.

    Sets up
        - logging,
        - default configuration values,
        - environment and
        - agent
    before starting the agent/environment interaction cycle by calling
    `interaction_loop`.

    If invalid arguments or options are given, it prints usage help information
    to the standard output and exits."""

    # Define some default configuration values.
    default_options = {}
    default_options["agent"] = "mc_aixi_ctw"
    default_options["agent-horizon"] = 5
    default_options["ct-depth"] = 30
    default_options["environment"] = "coin_flip"
    default_options["exploration"] = 0.0  # Do not explore.
    default_options["explore-decay"] = 1.0  # Exploration rate does not decay.
    default_options["learning-period"] = 0  # Learn forever.
    default_options["mc-simulations"] = 300
    default_options["profile"] = False  # Whether to profile code.
    default_options["terminate-age"] = 0  # Never die.
    default_options["verbose"] = False

    command_line_options = {}

    # Process the command line options and arguments.
    try:
        opts, args = getopt.gnu_getopt(
            argv,
            "d:e:h:l:m:o:pr:t:vx:",
            [
                "explore-decay=",
                "environment=",
                "agent-horizon=",
                "learning-period=",
                "mc-simulations=",
                "option",
                "profile",
                "terminate-age=",
                "ct-depth=",
                "verbose",
                "exploration=",
            ],
        )

        for opt, arg in opts:

            # TODO: elifs?
            if opt == "--help":
                usage()

            if opt in ("-d", "--explore-decay"):
                command_line_options["explore-decay"] = float(arg)
                continue

            if opt in ("-e", "--environment"):
                command_line_options["environment"] = str(arg)
                continue

            if opt in ("-h", "--agent-horizon"):
                command_line_options["agent-horizon"] = int(arg)
                continue

            if opt in ("-l", "--learning-period"):
                command_line_options["learning-period"] = int(arg)
                continue

            if opt in ("-m", "--mc-simulations"):
                command_line_options["mc-simulations"] = int(arg)
                continue

            if opt in ("-o", "--option"):
                # Split the associated argument into a key and value pair,
                # splitting on the '=' symbol.
                parts = arg.split("=")

                # Do we have enough parts to make a key=value pair?
                if len(parts) > 1:
                    key = parts[0].strip()
                    value = "=".join(parts[1:])
                    command_line_options[key] = value
                else:
                    # No. Show the usage, after printing an explanatory
                    # message.
                    print(
                        "Extra option '-o %s' is invalid. " % str(arg)
                        + "This needs to be in '-o key=value' format."
                        % str(arg)
                    )
                    usage()
                continue

            if opt in ("-p", "--profile"):
                command_line_options["profile"] = True
                continue

            if opt in ("-r", "--terminate-age"):
                command_line_options["terminate-age"] = int(arg)
                continue

            if opt in ("-t", "--ct-depth"):
                command_line_options["ct-depth"] = int(arg)
                continue

            if opt in ("-v", "--verbose"):
                command_line_options["verbose"] = True
                continue

            if opt in ("-x", "--exploration"):
                command_line_options["exploration"] = float(arg)
                continue

    except getopt.GetoptError as e:
        # We got an incorrect option. Show the usage and exit.
        usage()

    # Do we have any arguments left over?
    if len(args) > 0:
        # Yes. The first should be the name of a configuration file.
        filename = args[0]

        # Is this a valid filename?
        if not os.path.exists(filename):
            print(
                "Expected argument '%s' to be a configuration filename."
                % str(filename)
            )
            usage()

        # If we're here, we've got a valid filename.
        # Try reading it in as configuration file.
        config_contents = open(filename, "r").read()

        # Does the configuration contents contain an 'environment' section?
        if config_contents.find("[environment]") == -1:
            # No. Add one to the beginning.
            config_contents = "[environment]" + os.linesep + config_contents

        # Convert the contents into an in-memory file-like object, for parsing.
        config_stringio = StringIO.StringIO(config_contents)

        # Parse the given options, giving the default options as defaults to
        # the parser.
        config = configparser.RawConfigParser(default_options)

        # TODO: DeprecationWarning: This method will be removed in future
        #  versions.  Use 'parser.read_file()' instead.
        config.readfp(config_stringio)

        # Get the configuration options read in as a dictionary.
        # (This should exist in a section called 'environment'.)
        options = dict(config.items("environment"))
    else:
        # No. So set the options to be the default options.
        options = default_options

    # Let the command line options override the options read from the
    # configuration file--or
    # the default values--whichever way we got to this point.
    options.update(command_line_options)

    # Print the options we've received, if we've been requested to be verbose.
    verbose = bool(options.get("verbose", False))
    if verbose:
        for option_name, option_value in list(options.items()):
            print(
                "OPTION: '%s' = '%s'" % (str(option_name), str(option_value))
            )

    # Print an initial message header.
    message = (
        "cycle, observation, reward, action, explored, explore_rate, "
        "total reward, average reward, time, model size"
    )

    print(message)

    # Try to import an agent module with the given name.
    agent_name = options["agent"]

    # TODO: do I need to do this?
    # Ensure the name of the package we're trying to import has a prefix of
    # 'pyaixi.agents',
    # if it doesn't have one specified already.
    if agent_name.count(".") == 0:
        agent_package_name = "pyaixi.agents." + agent_name
    else:
        agent_package_name = agent_name

    try:
        agent_module = __import__(
            agent_package_name, globals(), locals(), [agent_name], 0
        )
    except Exception as e:
        # Exit with an error.
        sys.stderr.write(
            "ERROR: loading agent module '%s' caused error '%s'. "
            "Exiting." % (str(agent_name), str(e)) + os.linesep
        )
        sys.exit(1)

    # Find a subclass of the Agent class in the given module.
    agent_class = None
    agent_classname = ""
    for name in dir(agent_module):
        obj = getattr(agent_module, name)
        if inspect.isclass(obj) and "Agent" in [
            cls.__name__ for cls in obj.__bases__
        ]:
            agent_class = obj
            agent_classname = name
            break

    # Did we find a subclass of Agent?
    if agent_class is None:
        # No. Exit with an error.
        sys.stderr.write(
            "ERROR: agent module '%s' does not contain " % str(agent_name)
            + "a valid AIXI agent subclass. (Got '%s' instead.) "
            "Exiting." % str(agent_classname) + os.linesep
        )
        sys.exit(1)

    # Try to import an environment module with the given name.
    environment_name = options["environment"]

    # Ensure the name of the package we're trying to import has a prefix of
    # 'pyaixi.environments',
    # if it doesn't have one specified already.
    if environment_name.count(".") == 0:
        environment_package_name = "pyaixi.environments." + environment_name
    else:
        environment_package_name = environment_name

    try:
        environment_module = __import__(
            environment_package_name,
            globals(),
            locals(),
            [environment_name],
            0,
        )
    except Exception as e:
        # Exit with an error.
        sys.stderr.write(
            "ERROR: loading environment module '%s' caused error "
            "'%s'. Exiting." % (str(environment_name), str(e)) + os.linesep
        )
        sys.exit(1)

    # Find a subclass of the Environment class in the given module.
    environment_class = None
    environment_classname = ""

    for name, obj in inspect.getmembers(environment_module):
        if hasattr(obj, "__bases__") and "Environment" in [
            cls.__name__ for cls in obj.__bases__
        ]:
            environment_class = obj
            environment_classname = name
            break

    # Did we find a subclass of Environment?
    if environment_class is None:
        # No. Exit with an error.
        sys.stderr.write(
            "ERROR: environment module '%s' does not contain "
            % str(environment_name)
            + "a valid AIXI environment subclass. "
            "(Got '%s' instead.) Exiting." % str(environment_classname)
            + os.linesep
        )
        sys.exit(1)

    # Create an instance of the environment, using the discovered options.
    environment = environment_class(options=options)

    # Copy environment-dependent configuration options to the options.
    options["action-bits"] = environment.action_bits()
    options["observation-bits"] = environment.observation_bits()
    options["percept-bits"] = environment.percept_bits()
    options["reward-bits"] = environment.reward_bits()
    options["max-action"] = environment.maximum_action()
    options["max-observation"] = environment.maximum_observation()
    options["max-reward"] = environment.maximum_reward()

    # Set up the agent, using the created environment, and the updated options.
    agent = agent_class(environment=environment, options=options)

    # Run the main agent/environment interaction loop, profiling if requested
    # to do so.
    if bool(options.get("profile", False)):
        profile.runctx(
            "interaction_loop(agent = agent, "
            "environment = environment, "
            "options = options)",
            globals(),
            locals(),
        )
    else:
        interaction_loop(agent=agent, environment=environment, options=options)


# TODO: convert this to a multi-line string?
message = (
    "Usage: python aixi.py [-a | --agent <agent module name>"
    + os.linesep
    + "                      [-d | --explore-decay <exploration decay value, "
    "between 0 and 1>]"
    + os.linesep
    + "                      [-e | --environment <environment module name>]"
    + os.linesep
    + "                      [-h | --agent-horizon <search horizon>]"
    + os.linesep
    + "                      [-l | --learning-period <cycle count>]"
    + os.linesep
    + "                      [-m | --mc-simulations <number of simulations to "
    "run each step>]"
    + os.linesep
    + "                      [-o | --option <extra option name>=<value>]"
    + os.linesep
    + "                      [-p | --profile]"
    + os.linesep
    + "                      [-r | --terminate-age <number of cycles before "
    "stopping the run>]"
    + os.linesep
    + "                      [-t | --ct-depth <maximum depth of predicting "
    "context tree>]"
    + os.linesep
    + "                      [-x | --exploration <exploration factor, greater "
    "than 0>]"
    + os.linesep
    + "                      [-v | --verbose]"
    + os.linesep
    + "                      [<configuration file name to load>]"
    + os.linesep
    + os.linesep
)


def usage():
    """Prints usage information."""

    sys.stderr.write(message)
    sys.exit(2)


# Start the main function if this file has been executed, and not just
# imported.
if __name__ == "__main__":
    main(sys.argv[1:])
