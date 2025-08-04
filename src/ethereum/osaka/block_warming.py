"""
Block-level warming cost distribution for EIP-7928 (Block Access Lists).

This module implements the cost distribution mechanism for pre-warming storage
slots and accounts based on the block access list. The warming costs are 
distributed evenly among all transactions that access each resource.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

from ethereum_types.bytes import Bytes32
from ethereum_types.numeric import Uint

from .fork_types import Address
from .ssz_types import BlockAccessList
from .vm.gas import GAS_COLD_SLOAD, GAS_WARM_ACCESS, GAS_COLD_ACCOUNT_ACCESS

# Warming costs
STORAGE_WARMING_COST = GAS_COLD_SLOAD - GAS_WARM_ACCESS  # 2000 gas
ACCOUNT_WARMING_COST = GAS_COLD_ACCOUNT_ACCESS - GAS_WARM_ACCESS  # 2500 gas


@dataclass
class WarmingCostTracker:
    """Tracks which transactions access which storage slots and accounts."""
    storage_access_map: Dict[Tuple[Address, Bytes32], Set[int]] = field(default_factory=dict)
    account_access_map: Dict[Address, Set[int]] = field(default_factory=dict)


def build_warming_cost_tracker(
    block_access_list: BlockAccessList,
    num_transactions: int
) -> WarmingCostTracker:
    """
    Build a tracker that maps storage slots and accounts to the transactions that access them.
    
    Parameters
    ----------
    block_access_list :
        The block access list from the block.
    num_transactions :
        Number of transactions in the block.
        
    Returns
    -------
    tracker :
        A tracker containing the access mappings.
    """
    tracker = WarmingCostTracker()
    
    for account_changes in block_access_list.account_changes:
        address = account_changes.address
        
        # Track account accesses (all transactions that touch this account)
        tx_indices = set()
        for change in account_changes.balance_changes:
            if change.tx_index < num_transactions:
                tx_indices.add(change.tx_index)
        for change in account_changes.nonce_changes:
            if change.tx_index < num_transactions:
                tx_indices.add(change.tx_index)
        for change in account_changes.code_changes:
            if change.tx_index < num_transactions:
                tx_indices.add(change.tx_index)
        for storage_change in account_changes.storage_changes:
            if storage_change.tx_index < num_transactions:
                tx_indices.add(storage_change.tx_index)
        
        if tx_indices:
            tracker.account_access_map[address] = tx_indices
        
        # Track storage slot accesses
        for storage_change in account_changes.storage_changes:
            if storage_change.tx_index < num_transactions:
                key = (address, Bytes32(storage_change.slot))
                if key not in tracker.storage_access_map:
                    tracker.storage_access_map[key] = set()
                tracker.storage_access_map[key].add(storage_change.tx_index)
    
    return tracker


def calculate_warming_costs(tracker: WarmingCostTracker) -> Dict[Address, Uint]:
    """
    Calculate the total warming cost for each address.
    
    Parameters
    ----------
    tracker :
        The warming cost tracker.
        
    Returns
    -------
    warming_costs :
        Total warming cost for each address.
    """
    warming_costs: Dict[Address, Uint] = {}
    
    # Add account warming costs
    for address in tracker.account_access_map:
        if address not in warming_costs:
            warming_costs[address] = Uint(0)
        warming_costs[address] += ACCOUNT_WARMING_COST
    
    # Add storage warming costs
    for (address, _) in tracker.storage_access_map:
        if address not in warming_costs:
            warming_costs[address] = Uint(0)
        warming_costs[address] += STORAGE_WARMING_COST
    
    return warming_costs


def calculate_transaction_warming_deltas(
    tracker: WarmingCostTracker,
    warming_costs: Dict[Address, Uint]
) -> Dict[int, Uint]:
    """
    Calculate how much each transaction should pay for warming.
    
    The warming cost for each resource is split evenly among all transactions
    that access that resource.
    
    Parameters
    ----------
    tracker :
        The warming cost tracker.
    warming_costs :
        Total warming costs per address.
        
    Returns
    -------
    tx_warming_deltas :
        Warming cost delta for each transaction index.
    """
    tx_warming_costs: Dict[int, Uint] = {}
    
    # Distribute account warming costs
    for address, accessing_txs in tracker.account_access_map.items():
        if len(accessing_txs) > 0:
            cost_per_tx = ACCOUNT_WARMING_COST // Uint(len(accessing_txs))
            remainder = ACCOUNT_WARMING_COST % Uint(len(accessing_txs))
            
            for i, tx_idx in enumerate(sorted(accessing_txs)):
                if tx_idx not in tx_warming_costs:
                    tx_warming_costs[tx_idx] = Uint(0)
                tx_warming_costs[tx_idx] += cost_per_tx
                # Distribute remainder to first transactions
                if i < remainder:
                    tx_warming_costs[tx_idx] += Uint(1)
    
    # Distribute storage warming costs
    for (address, storage_key), accessing_txs in tracker.storage_access_map.items():
        if len(accessing_txs) > 0:
            cost_per_tx = STORAGE_WARMING_COST // Uint(len(accessing_txs))
            remainder = STORAGE_WARMING_COST % Uint(len(accessing_txs))
            
            for i, tx_idx in enumerate(sorted(accessing_txs)):
                if tx_idx not in tx_warming_costs:
                    tx_warming_costs[tx_idx] = Uint(0)
                tx_warming_costs[tx_idx] += cost_per_tx
                # Distribute remainder to first transactions
                if i < remainder:
                    tx_warming_costs[tx_idx] += Uint(1)
    
    return tx_warming_costs