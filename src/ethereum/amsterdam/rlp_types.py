"""
RLP Types for EIP-7928 Block-Level Access Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module defines the RLP data structures for Block-Level Access Lists
as specified in EIP-7928. These structures enable efficient encoding and
decoding of all accounts and storage locations accessed during block execution.

The encoding follows the pattern:
address -> field -> block_access_index -> change
"""

from dataclasses import dataclass
from typing import Tuple

from ethereum_types.bytes import Bytes, Bytes20, Bytes32
from ethereum_types.frozen import slotted_freezable
from ethereum_types.numeric import U64, U256, Uint

# Type aliases for clarity (matching EIP-7928 specification)
Address = Bytes20
StorageKey = Bytes32
StorageValue = Bytes32
CodeData = Bytes
BlockAccessIndex = Uint  # uint16 in the spec, but using Uint for compatibility
Balance = U256  # Post-transaction balance in wei
Nonce = U64

# Constants chosen to support a 630m block gas limit
MAX_TXS = 30_000
# MAX_SLOTS = 300_000
# MAX_ACCOUNTS = 300_000
MAX_CODE_SIZE = 24_576
MAX_CODE_CHANGES = 1


@slotted_freezable
@dataclass
class StorageChange:
    """
    Storage change: [block_access_index, new_value]
    RLP encoded as a list
    """

    block_access_index: BlockAccessIndex
    new_value: StorageValue


@slotted_freezable
@dataclass
class BalanceChange:
    """
    Balance change: [block_access_index, post_balance]
    RLP encoded as a list
    """

    block_access_index: BlockAccessIndex
    post_balance: Balance


@slotted_freezable
@dataclass
class NonceChange:
    """
    Nonce change: [block_access_index, new_nonce]
    RLP encoded as a list
    """

    block_access_index: BlockAccessIndex
    new_nonce: Nonce


@slotted_freezable
@dataclass
class CodeChange:
    """
    Code change: [block_access_index, new_code]
    RLP encoded as a list
    """

    block_access_index: BlockAccessIndex
    new_code: CodeData


@slotted_freezable
@dataclass
class SlotChanges:
    """
    All changes to a single storage slot: [slot, [changes]]
    RLP encoded as a list
    """

    slot: StorageKey
    changes: Tuple[StorageChange, ...]


@slotted_freezable
@dataclass
class SlotReads:
    """
    Storage slot reads: [slot, [block_access_indices]]
    RLP encoded as a list
    Only includes slots that were read but not written to
    """

    slot: StorageKey
    block_access_indices: Tuple[BlockAccessIndex, ...]


@slotted_freezable
@dataclass
class AccountRead:
    """
    Account read-only access: [block_access_index]
    Used for BALANCE, EXTCODEHASH, etc. operations
    Only includes tx indices where account was accessed but not modified
    """

    block_access_index: BlockAccessIndex


@slotted_freezable
@dataclass
class AccountChanges:
    """
    All changes and reads for a single account, grouped by field type.
    This eliminates address redundancy across different change types.
    Now includes tx indices for all read operations.
    RLP encoded as: [address, storage_changes, storage_reads,
    balance_changes, nonce_changes, code_changes, account_reads]
    """

    address: Address

    # Storage changes (slot -> [block_access_index -> new_value])
    storage_changes: Tuple[SlotChanges, ...]

    # Storage reads (slot -> [block_access_index]) - only for slots not written to
    storage_reads: Tuple[SlotReads, ...]

    # Balance changes ([block_access_index -> post_balance])
    balance_changes: Tuple[BalanceChange, ...]

    # Nonce changes ([block_access_index -> new_nonce])
    nonce_changes: Tuple[NonceChange, ...]

    # Code changes ([block_access_index -> new_code]) - typically 0 or 1
    code_changes: Tuple[CodeChange, ...]

    # Account read-only accesses ([block_access_index]) - for BALANCE, EXTCODEHASH, etc.
    # Only includes tx indices where account was accessed but not modified
    account_reads: Tuple[AccountRead, ...]


@slotted_freezable
@dataclass
class BlockAccessList:
    """
    Block-Level Access List for EIP-7928.
    Contains all addresses accessed during block execution.
    RLP encoded as a list of AccountChanges
    """

    account_changes: Tuple[AccountChanges, ...]
