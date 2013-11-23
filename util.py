#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define some helper functions.
"""

from __future__ import division
from __future__ import print_function

try:
    import collections
    IterableUserDict = collections.UserDict
except:
    import UserDict
    IterableUserDict = UserDict.IterableUserDict
# end try

def bits_required(integer_value):
    """ Return the number of bits required to store the given integer.
    """
    assert type(integer_value) == int and integer_value >= 0, "The given number must be an integer greater than or equal to zero."

    # The bin built-in function converts the value to a binary string, but adds an '0b' prefix.
    # Count the length of the string, but subtract 2 for the '0b' prefix.
    return len(bin(integer_value)) - 2
# end def

def decode(symbol_list, bit_count):
    """ Decodes the value encoded on the end of a list of symbols.
        Each symbol is a bit in the binary representation of the value, with more significant
        bits at the end of the list.

        - `symbol_list` - the list of symbols to decode from.
        - `bit_count` - the number of bits from the end of the symbol list to decode.
    """
    assert bit_count > 0, "The given number of bits (%d) is invalid." % bit_count
    assert bit_count <= len(symbol_list), "The given number of bits (%d) is greater than the length of the symbol list. (%d)" % (bit_count, len(symbol_list))

    # Take the last `bit_count` number of symbols from the end of the given symbol list.
    bits = symbol_list[-bit_count:]

    # Reverse the list of bits, and make a string out of them.
    bits.reverse()
    bit_string = ''.join(map(str, bits))

    # Return the bit string as an integer via the built-in int command, telling it that the number in the string is binary/base 2.
    return int(bit_string, 2)
# end def

def encode(integer_symbol, bit_count):
    """ Returns an updated version of the given symbol list with the given symbol encoded into binary.

        - `symbol_list` - the list onto which to encode the value.
        - `integer_symbol` - the integer value to be encoded.
        - `bit_count` - the number of bits from the end of the symbol list to decode.
    """

    assert type(integer_symbol) == int and integer_symbol >= 0, "The given symbol must be an integer greater than or equal to zero."

    # Convert the symbol into a bit string.
    bit_string = bin(integer_symbol)

    # Strip off any '0b' prefix.
    if bit_string.startswith('0b'):
        bit_string = bit_string[2:]
    # end if

    # Convert the string into a list of integers.
    bits = [int(bit) for bit in list(bit_string)]

    # Check that the number of bits is not bigger than the given bit count.
    bits_length = len(bits)
    assert bit_count >= bits_length, \
           "The given number of bits %d to encode is smaller than the bits needed to encode %d." % \
               (bit_count, bits_length)

    # Calculate how many bits we need to pad the bit string with, if any, and pad with zeros.
    pad_list = [0 for i in xrange(0, bits_length - bit_count)]

    # Return the newly created bit list, with the zero padding first.
    symbol_list = pad_list + bits
    return symbol_list
# end def

def enum(*sequential, **named):
    """ Define an enumeration type helper, since the operation of this codebase depends heavily on enumeration types.

        Usage:

        new_enum = enum('value1', 'value2')

        This code is base on code by StackOverflow user Alec Thomas http://stackoverflow.com/users/7980/alec-thomas
        From: http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
        License: CC-Wiki/CC BY-SA 3.0 with attribution.
    """

    # Construct a mapping of the sequential values to the names.
    enum_dict = dict(zip([str(value) for value in sequential], range(len(sequential))), **named)

    # Reverse this, so that we've got a way to quickly look up values to names.
    reverse = dict((value, key) for key, value in enum_dict.items())

    # Set up a dictionary (with user-modifiable attributes) from the reverse mapping,
    # so that iteration over the enumeration and membership checks are possible.
    enums = IterableUserDict(reverse)

    # Add the original and reverse mappings to the dictionary.
    enums.mapping = enum_dict
    enums.reverse_mapping = reverse

    # Make each of the name values have an attribute, for convenience.
    # e.g. new_enum.value1 == 0   new_enum.value2 == 1
    for (key, value) in enum_dict.items():
        setattr(enums, str(key), int(value))
    # end for

    # Return the generated structure.
    return enums
# end def