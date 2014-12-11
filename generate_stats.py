#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script for generating statistics from the read input, outputting a statistic assessment to standard output.

This works around the current lack of Numpy/Scipy support in the PyPy and Jython interpreters.

Usage: python generate_stats.py < <original and comparison results separated and terminated by a single empty line>
"""

from __future__ import division
from __future__ import unicode_literals

import os
import math
import sys

try:
    from scipy import stats
    scipy_available = True
except:
    scipy_available = False
# end try

def do_hypothesis_tests(original, comparison):
    """ Does hypothesis tests on the two given lists of results, and returns a summary in a string for later display.
    """

    result = ""

    # Get the medians of each result group, for use in later comparisons.
    try:
        original_median = stats.nanmedian(original)
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while getting median of original results: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    try:
        comparison_median = stats.nanmedian(comparison)
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while getting median of comparison results: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    # Describe both original and comparison result lists.
    try:
        original_count, (original_min, original_max), original_mean, original_var, original_skew, original_kurtosis = \
            stats.describe(original)
        result += "Original results:   count=%d min=%.2f max=%.2f mean=%.2f median=%.2f variance=%.2f skewness=%.2f kurtosis=%.2f" % \
                  (original_count, original_min, original_max, original_mean, original_median, original_var,
                   original_skew, original_kurtosis) + os.linesep + os.linesep
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while describing original results: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    try:
        comparison_count, (comparison_min, comparison_max), comparison_mean, comparison_var, comparison_skew, comparison_kurtosis = \
            stats.describe(comparison)
        result += "Comparison results: count=%d min=%.2f max=%.2f mean=%.2f median=%.2f variance=%.2f skewness=%.2f kurtosis=%.2f" % \
                  (comparison_count, comparison_min, comparison_max, comparison_mean, comparison_median, comparison_var,
                   comparison_skew, comparison_kurtosis) + os.linesep + os.linesep
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while describing comparison results: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    # Perform a Fligner's median test to see if the rewards have the same variance, assuming nothing about the population.
    try:
        f_median_chi_squared, f_median_p_value = stats.fligner(original, comparison, center = 'median')
        result += "Fligner's median test - chi-squared = %.2f, p(equal variance) = %.2f%%" % \
                  (f_median_chi_squared, f_median_p_value * 100) + os.linesep + os.linesep
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while performing a Fligner's median test: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    # Perform a Fligner's mean test to see if the rewards have the same variance, assuming the populations are symmetrical.
    try:
        f_mean_chi_squared, f_mean_p_value = stats.fligner(original, comparison, center = 'mean')
        result += "Fligner's mean test - chi-squared = %.2f, p(equal variance, symmetric distribution) = %.2f%%" % \
                  (f_mean_chi_squared, f_mean_p_value * 100) + os.linesep + os.linesep
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while performing a Fligner's mean test: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    # Perform Mood's test to see if the rewards have the same scale parameter.
    try:
        m_z_score, m_p_value = stats.mood(original, comparison)
        result += "Mood's median test - z-score = %.2f, p(equal scale factor) = %.2f%%" % \
                  (m_z_score, m_p_value * 100) + os.linesep + os.linesep
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while performing a Mood's median test: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    # Perform a Mann-Whitney U test to see if the rewards in the original are greater than the comparison, assuming
    # nothing about the population.
    try:
        mw_u_statistic, mw_p_value = stats.mannwhitneyu(original, comparison)
        result += "Mann-Whitney U test - U-statistic = %.2f, p(original = comparison) = %.2f%%" % \
                   (mw_u_statistic, mw_p_value * 2 * 100) + os.linesep + os.linesep
    except Exception:
        e = sys.exc_info()[1]
        result += "Got error while performing a Mann-Whitney U test: '%s'" % str(e) + os.linesep + os.linesep
    # end try

    # Return the string describing the results.
    return result
# end def

def main(argv):
    """ Entry point of the program. Reads in the input, processes the input,
        and reports the results back to the caller.
    """

    # Check for scipy support.
    if not scipy_available:
        message = "ERROR: Scipy is unavailable to generate statistics." + os.linesep + \
                  "Please install Scipy if you wish to generate statistics." + os.linesep
        sys.stderr.write(message)
        exit(1)
    # end if

    # If we're here, read in any input.
    original = []
    comparison = []
    on_original_list = True
    for line in sys.stdin:
        # See if we've got a valid line of input.
        line = line.strip()
        if len(line) > 0:
            # We do. Are we still on the original list of results?
            if on_original_list:
                # Yes. Put this value into the original list of results as a float.
                original.append(float(line))
            else:
                # No. Put this value into the comparison list of results as a float.
                comparison.append(float(line))
            # end if
        else:
            # No, we don't. Treat this as the end of a list of results.
            # Are we on the original list of results?
            if on_original_list:
                # Yes. Move on to the next list of results.
                on_original_list = False
            else:
                # No. This is the end of the input stream.
                break
            # end if
        # end if
    # end for

    # Consume any remaining input.
    sys.stdin.readlines()

    # Check the given lists. If they're empty, report an error.
    len_original = len(original)
    len_comparison = len(comparison)
    if len_original == 0 or len_comparison == 0:
        message = "ERROR: original and comparsion result lists not passed correctly."
        sys.stderr.write(message)
        exit(1)
    # end if

    # Pass the result lists to a function that generates a hypothesis testing summary.
    result = do_hypothesis_tests(original, comparison)

    sys.stdout.write(result)
# end def

# Start the main function if this file has been executed, and not just imported.
if __name__ == "__main__":
    main(sys.argv[1:])
# end def