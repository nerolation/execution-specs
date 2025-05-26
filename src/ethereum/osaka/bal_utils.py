"""
Utility functions for EIP-7928 Block Access Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Helper functions for encoding and tracking BAL data.
"""

from typing import Union

from ethereum_types.bytes import Bytes
from ethereum_types.numeric import U256, Uint


def encode_balance_delta(delta: Union[int, Uint]) -> Bytes:
    """
    Encode a balance delta as a 12-byte signed integer using two's complement.
    
    Parameters
    ----------
    delta :
        The balance change (positive or negative).
        
    Returns
    -------
    encoded : Bytes
        12-byte encoded balance delta.
    """
    if isinstance(delta, Uint):
        # Positive value - direct encoding
        value = int(delta)
        is_negative = False
    else:
        # Handle signed integer
        value = delta
        is_negative = value < 0
        if is_negative:
            value = abs(value)
    
    # Convert to bytes
    if is_negative:
        # Two's complement for negative values
        # Maximum value for 12 bytes is 2^96 - 1
        max_val = (1 << 96) - 1
        encoded_val = max_val - value + 1
    else:
        encoded_val = value
    
    # Convert to 12 bytes
    return encoded_val.to_bytes(12, byteorder='big')


def decode_balance_delta(encoded: Bytes) -> int:
    """
    Decode a 12-byte signed integer from two's complement.
    
    Parameters
    ----------
    encoded :
        12-byte encoded balance delta.
        
    Returns
    -------
    delta : int
        The decoded balance change.
    """
    if len(encoded) != 12:
        raise ValueError("Balance delta must be 12 bytes")
    
    value = int.from_bytes(encoded, byteorder='big')
    
    # Check if negative (MSB is 1)
    if value >> 95:  # Check the sign bit for 96-bit number
        # Two's complement - convert back to negative
        max_val = (1 << 96)
        return value - max_val
    else:
        return value


def compute_balance_delta(old_balance: U256, new_balance: U256) -> Bytes:
    """
    Compute the balance delta between two balances.
    
    Parameters
    ----------
    old_balance :
        The balance before the change.
    new_balance :
        The balance after the change.
        
    Returns
    -------
    delta : Bytes
        12-byte encoded balance delta.
    """
    old = int(old_balance)
    new = int(new_balance)
    delta = new - old
    return encode_balance_delta(delta) 