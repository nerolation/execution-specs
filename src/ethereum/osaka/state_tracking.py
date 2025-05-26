"""
State Tracking Wrappers for EIP-7928
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Wrapper functions that add BAL tracking to state operations.
"""

from typing import Optional

from ethereum_types.bytes import Bytes, Bytes32
from ethereum_types.numeric import U256, Uint

from .fork_types import Address
from .state import (
    State,
    TransientStorage,
    get_account,
    get_storage as _get_storage,
    get_transient_storage as _get_transient_storage,
    increment_nonce as _increment_nonce,
    move_ether as _move_ether,
    set_account_balance as _set_account_balance,
    set_code as _set_code,
    set_storage as _set_storage,
    set_transient_storage as _set_transient_storage,
)
from .vm import BlockEnvironment, Evm


def track_account_access_with_bal(evm: Evm, address: Address) -> None:
    """
    Track that an account was accessed for BAL.
    
    This should be called by opcodes that access account information
    like BALANCE, EXTCODESIZE, EXTCODECOPY, EXTCODEHASH.
    
    Parameters
    ----------
    evm :
        The current EVM frame.
    address :
        Address of the account being accessed.
    """
    if evm.message.block_env.bal_tracker:
        evm.message.block_env.bal_tracker.track_account_access(address)


def get_storage_with_tracking(
    evm: Evm, address: Address, key: Bytes32
) -> U256:
    """
    Get storage value and track the read access for BAL.
    
    Parameters
    ----------
    evm :
        The current EVM frame.
    address :
        Address of the account.
    key :
        Storage key to read.
        
    Returns
    -------
    value : U256
        The storage value.
    """
    value = _get_storage(evm.message.block_env.state, address, key)
    
    # Track the read access
    if evm.message.block_env.bal_tracker:
        evm.message.block_env.bal_tracker.track_storage_read(
            evm.message.block_env.state, address, key
        )
    
    return value


def set_storage_with_tracking(
    evm: Evm, address: Address, key: Bytes32, value: U256
) -> None:
    """
    Set storage value and track the write access for BAL.
    
    Parameters
    ----------
    evm :
        The current EVM frame.
    address :
        Address of the account.
    key :
        Storage key to write.
    value :
        Value to write.
    """
    _set_storage(evm.message.block_env.state, address, key, value)
    
    # Track the write access with the final value
    if evm.message.block_env.bal_tracker:
        evm.message.block_env.bal_tracker.track_storage_write(
            evm.message.block_env.state, address, key, value.to_be_bytes32()
        )


def move_ether_with_tracking(
    block_env: BlockEnvironment,
    sender_address: Address,
    recipient_address: Address,
    amount: U256,
) -> None:
    """
    Move ether between accounts and track balance changes for BAL.
    
    Parameters
    ----------
    block_env :
        The block environment containing state and BAL tracker.
    sender_address :
        Address sending ether.
    recipient_address :
        Address receiving ether.
    amount :
        Amount of ether to transfer.
    """
    if block_env.bal_tracker:
        # Track sender balance change
        sender_balance = get_account(block_env.state, sender_address).balance
        new_sender_balance = sender_balance - amount
        block_env.bal_tracker.track_balance_change(
            block_env.state, sender_address, sender_balance, new_sender_balance
        )
        
        # Track recipient balance change
        recipient_balance = get_account(block_env.state, recipient_address).balance
        new_recipient_balance = recipient_balance + amount
        block_env.bal_tracker.track_balance_change(
            block_env.state, recipient_address, recipient_balance, new_recipient_balance
        )
    
    _move_ether(block_env.state, sender_address, recipient_address, amount)


def set_code_with_tracking(
    block_env: BlockEnvironment, address: Address, code: Bytes
) -> None:
    """
    Set account code and track the deployment for BAL.
    
    Parameters
    ----------
    block_env :
        The block environment containing state and BAL tracker.
    address :
        Address of the account.
    code :
        Bytecode to set.
    """
    _set_code(block_env.state, address, code)
    
    # Track code deployment
    if block_env.bal_tracker and code:
        block_env.bal_tracker.track_code_deployment(address, code)


def increment_nonce_with_tracking(
    block_env: BlockEnvironment, address: Address
) -> None:
    """
    Increment nonce and track the change for BAL.
    
    Parameters
    ----------
    block_env :
        The block environment containing state and BAL tracker.
    address :
        Address whose nonce is being incremented.
    """
    if block_env.bal_tracker:
        account = get_account(block_env.state, address)
        old_nonce = account.nonce
        new_nonce = old_nonce + Uint(1)
        block_env.bal_tracker.track_nonce_change(
            block_env.state, address, U64(old_nonce), U64(new_nonce)
        )
    
    _increment_nonce(block_env.state, address) 