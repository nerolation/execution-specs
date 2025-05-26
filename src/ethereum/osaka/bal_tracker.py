"""
EIP-7928 Block Access List Tracker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Tracks state accesses and changes during block execution for EIP-7928.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ethereum_types.bytes import Bytes
from ethereum_types.numeric import U16, U64, U256, Uint

from .bal_types import StorageKey, StorageValue, TxIndex
from .bal_utils import compute_balance_delta
from .fork_types import Address
from .state import State, get_account, get_storage


@dataclass
class BalTracker:
    """
    Tracks all state accesses and changes for BAL generation.
    """
    
    # Current transaction index
    current_tx_index: Optional[TxIndex] = None
    
    # Storage accesses: (address, slot) -> [(tx_index, is_write, value)]
    storage_accesses: Dict[Tuple[Address, StorageKey], List[Tuple[TxIndex, bool, StorageValue]]] = field(
        default_factory=dict
    )
    
    # Balance tracking: address -> initial balance before any transaction touched it
    initial_balances: Dict[Address, U256] = field(default_factory=dict)
    
    # Per-transaction balance tracking: (tx_index, address) -> (initial, current)
    tx_balance_tracking: Dict[Tuple[TxIndex, Address], Tuple[U256, U256]] = field(
        default_factory=dict
    )
    
    # Final balance changes: address -> [(tx_index, delta)]
    balance_changes: Dict[Address, List[Tuple[TxIndex, Bytes]]] = field(
        default_factory=dict
    )
    
    # Nonce tracking: address -> nonce before first change
    initial_nonces: Dict[Address, U64] = field(default_factory=dict)
    
    # Code deployments: address -> code
    code_deployments: Dict[Address, Bytes] = field(default_factory=dict)
    
    # Accessed addresses (for validation)
    accessed_addresses: Set[Address] = field(default_factory=set)
    
    def start_transaction(self, tx_index: Uint) -> None:
        """Start tracking a new transaction."""
        self.current_tx_index = TxIndex(tx_index)
    
    def end_transaction(self) -> None:
        """
        End tracking the current transaction.
        Finalize balance changes - only record if net change occurred.
        """
        if self.current_tx_index is None:
            return
            
        # Process all balance changes for this transaction
        tx_index = self.current_tx_index
        addresses_to_check = [
            addr for (tx_idx, addr), _ in self.tx_balance_tracking.items()
            if tx_idx == tx_index
        ]
        
        for address in addresses_to_check:
            initial, final = self.tx_balance_tracking[(tx_index, address)]
            
            # Only record if there was a net change
            if initial != final:
                delta = compute_balance_delta(initial, final)
                
                if address not in self.balance_changes:
                    self.balance_changes[address] = []
                
                self.balance_changes[address].append((tx_index, delta))
        
        self.current_tx_index = None
    
    def track_storage_read(
        self, state: State, address: Address, slot: StorageKey
    ) -> None:
        """Track a storage read operation."""
        if self.current_tx_index is None:
            return  # System transaction, not tracked
            
        key = (address, slot)
        if key not in self.storage_accesses:
            self.storage_accesses[key] = []
        
        # For reads, we don't store the value in BAL
        # Just mark that this slot was accessed
        self.accessed_addresses.add(address)
    
    def track_storage_write(
        self, state: State, address: Address, slot: StorageKey, value: StorageValue
    ) -> None:
        """Track a storage write operation."""
        if self.current_tx_index is None:
            return  # System transaction, not tracked
            
        key = (address, slot)
        if key not in self.storage_accesses:
            self.storage_accesses[key] = []
        
        # Record the write with the final value
        self.storage_accesses[key].append(
            (self.current_tx_index, True, value)
        )
        self.accessed_addresses.add(address)
    
    def track_balance_change(
        self, state: State, address: Address, old_balance: U256, new_balance: U256
    ) -> None:
        """Track a balance change."""
        if self.current_tx_index is None:
            return  # System transaction, not tracked
            
        # Store the very first balance ever seen for this address
        if address not in self.initial_balances:
            self.initial_balances[address] = old_balance
        
        tx_key = (self.current_tx_index, address)
        
        # If this is the first time we see this address in this transaction,
        # record its initial balance for this transaction
        if tx_key not in self.tx_balance_tracking:
            self.tx_balance_tracking[tx_key] = (old_balance, new_balance)
        else:
            # Update the current balance, keeping the initial
            initial, _ = self.tx_balance_tracking[tx_key]
            self.tx_balance_tracking[tx_key] = (initial, new_balance)
    
    def track_nonce_change(
        self, state: State, address: Address, old_nonce: U64, new_nonce: U64
    ) -> None:
        """
        Track a nonce change for CREATE/CREATE2 deployer accounts.
        
        This is only called when an account deploys a contract using CREATE or CREATE2.
        Regular transaction nonce increments are NOT tracked.
        
        Parameters
        ----------
        state :
            The current state.
        address :
            Address of the deployer account.
        old_nonce :
            Nonce before the CREATE/CREATE2 operation.
        new_nonce :
            Nonce after the CREATE/CREATE2 operation.
        """
        if self.current_tx_index is None:
            return  # System transaction, not tracked
            
        # Store initial nonce on first change
        if address not in self.initial_nonces and old_nonce != new_nonce:
            self.initial_nonces[address] = old_nonce
    
    def track_code_deployment(
        self, address: Address, code: Bytes
    ) -> None:
        """Track a contract deployment."""
        if self.current_tx_index is None:
            return  # System transaction, not tracked
            
        self.code_deployments[address] = code
        self.accessed_addresses.add(address)
    
    def track_account_access(self, address: Address) -> None:
        """Track that an account was accessed."""
        if self.current_tx_index is None:
            return  # System transaction, not tracked
            
        self.accessed_addresses.add(address) 