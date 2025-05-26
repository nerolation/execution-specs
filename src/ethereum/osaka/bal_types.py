"""
Block Access List (BAL) Types for EIP-7928
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Types for block-level access lists that enable parallel transaction execution.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from ethereum_types.bytes import Bytes
from ethereum_types.numeric import U16, U64, Uint

from .fork_types import Address

# Type aliases
StorageKey = Bytes  # 32 bytes
StorageValue = Bytes  # 32 bytes
TxIndex = U16
Nonce = U64
BalanceDelta = Bytes  # 12 bytes, signed two's complement

# Constants for EIP-7928 - chosen to support a 630m block gas limit
MAX_TXS = 30_000
MAX_SLOTS = 300_000
MAX_ACCOUNTS = 300_000
MAX_CODE_SIZE = 24_576  # Maximum contract bytecode size in bytes


@dataclass
class PerTxAccess:
    """
    Access information for a single transaction.
    """
    tx_index: TxIndex
    value_after: StorageValue  # value in state after the last access


@dataclass
class SlotAccess:
    """
    Access information for a storage slot.
    """
    slot: StorageKey
    accesses: List[PerTxAccess]  # empty for reads


@dataclass
class AccountAccess:
    """
    Access information for an account.
    """
    address: Address
    accesses: List[SlotAccess]
    code: Optional[Bytes]  # Optional field for contract bytecode


# The main block access list type
BlockAccessList = List[AccountAccess]


@dataclass
class BalanceChange:
    """
    Balance change for a single transaction.
    """
    tx_index: TxIndex
    delta: BalanceDelta  # signed integer, encoded as 12-byte vector


@dataclass
class AccountBalanceDiff:
    """
    Balance changes for an account across all transactions in a block.
    """
    address: Address
    changes: List[BalanceChange]


# Balance diffs for all accounts in a block
BalanceDiffs = List[AccountBalanceDiff]


@dataclass
class AccountNonce:
    """
    Nonce information for accounts that deployed contracts.
    """
    address: Address  # account address
    nonce_before: Nonce  # nonce value before the transaction


# Nonce diffs for all CREATE/CREATE2 operations in a block
NonceDiffs = List[AccountNonce] 