"""
Ethereum Specification
^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Entry point for the Ethereum specification.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from ethereum_types.bytes import Bytes, Bytes32
from ethereum_types.numeric import U64, U256, Uint

from ethereum.crypto.hash import Hash32, keccak256
from ethereum.exceptions import InvalidBlock, InvalidSenderError

from .. import rlp
from . import vm
from .blocks import Block, Header, Log, Receipt, Withdrawal
from .bloom import logs_bloom
from .fork_types import Address, Bloom, Root, VersionedHash
from .state import (
    State,
    TransientStorage,
    account_exists_and_is_empty,
    destroy_account,
    destroy_touched_empty_accounts,
    get_account,
    increment_nonce,
    process_withdrawal,
    set_account_balance,
    state_root,
)
from .transactions import (
    AccessListTransaction,
    BlobTransaction,
    FeeMarketTransaction,
    LegacyTransaction,
    Transaction,
    calculate_intrinsic_cost,
    decode_transaction,
    encode_transaction,
    recover_sender,
    validate_transaction,
)
from .trie import Trie, root, trie_set
from .utils.hexadecimal import hex_to_address
from .utils.message import prepare_message
from .vm import Message
from .vm.gas import (
    calculate_blob_gas_price,
    calculate_data_fee,
    calculate_excess_blob_gas,
    calculate_total_blob_gas,
)
from .vm.interpreter import process_message_call

BASE_FEE_MAX_CHANGE_DENOMINATOR = Uint(8)
ELASTICITY_MULTIPLIER = Uint(2)
GAS_LIMIT_ADJUSTMENT_FACTOR = Uint(1024)
GAS_LIMIT_MINIMUM = Uint(5000)
EMPTY_OMMER_HASH = keccak256(rlp.encode([]))
SYSTEM_ADDRESS = hex_to_address("0xfffffffffffffffffffffffffffffffffffffffe")
BEACON_ROOTS_ADDRESS = hex_to_address(
    "0x000F3df6D732807Ef1319fB7B8bB8522d0Beac02"
)
SYSTEM_TRANSACTION_GAS = Uint(30000000)
MAX_BLOB_GAS_PER_BLOCK = Uint(786432)
VERSIONED_HASH_VERSION_KZG = b"\x01"


@dataclass
class BlockChain:
    """
    History and current state of the block chain.
    """

    blocks: List[Block]
    state: State
    chain_id: U64
    last_block_gas_used: Uint
    last_receipt_root: Root
    last_block_logs_bloom: Bloom
    


def apply_fork(old: BlockChain) -> BlockChain:
    """
    Transforms the state from the previous hard fork (`old`) into the block
    chain object for this hard fork and returns it.

    When forks need to implement an irregular state transition, this function
    is used to handle the irregularity. See the :ref:`DAO Fork <dao-fork>` for
    an example.

    Parameters
    ----------
    old :
        Previous block chain object.

    Returns
    -------
    new : `BlockChain`
        Upgraded block chain object for this hard fork.
    """
    return old


def get_last_256_block_hashes(chain: BlockChain) -> List[Hash32]:
    """
    Obtain the list of hashes of the previous 256 blocks in order of
    increasing block number.

    This function will return less hashes for the first 256 blocks.

    The ``BLOCKHASH`` opcode needs to access the latest hashes on the chain,
    therefore this function retrieves them.

    Parameters
    ----------
    chain :
        History and current state.

    Returns
    -------
    recent_block_hashes : `List[Hash32]`
        Hashes of the recent 256 blocks in order of increasing block number.
    """
    recent_blocks = chain.blocks[-255:]
    # TODO: This function has not been tested rigorously
    if len(recent_blocks) == 0:
        return []

    recent_block_hashes = []

    for block in recent_blocks:
        prev_block_hash = block.header.parent_hash
        recent_block_hashes.append(prev_block_hash)

    # We are computing the hash only for the most recent block and not for
    # the rest of the blocks as they have successors which have the hash of
    # the current block as parent hash.
    most_recent_block_hash = keccak256(rlp.encode(recent_blocks[-1].header))
    recent_block_hashes.append(most_recent_block_hash)

    return recent_block_hashes


def state_transition(chain: BlockChain, block: Block) -> None:
    """
    Attempts to apply a block to an existing block chain.

    All parts of the block's contents need to be verified before being added
    to the chain. Blocks are verified by ensuring that the contents of the
    block make logical sense with the contents of the parent block. The
    information in the block's header must also match the corresponding
    information in the block.

    To implement Ethereum, in theory clients are only required to store the
    most recent 255 blocks of the chain since as far as execution is
    concerned, only those blocks are accessed. Practically, however, clients
    should store more blocks to handle reorgs.

    Parameters
    ----------
    chain :
        History and current state.
    block :
        Block to apply to `chain`.
    """  
    parent_header = chain.blocks[-1].header
    validate_header(block.header, parent_header)

    (
        sender_addresses, 
        coinbase_balance_after_upfront_cost,
        total_intrinsic_gas,
    ) = check_block_static(chain, block)
    set_account_balance(chain.state, block.header.coinbase, U256(coinbase_balance_after_upfront_cost))

    apply_body_output = apply_body(
        chain.state,
        get_last_256_block_hashes(chain),
        block.header.coinbase,
        block.header.number,
        block.header.base_fee_per_gas,
        block.header.gas_limit,
        block.header.timestamp,
        block.header.prev_randao,
        block.transactions,
        chain.chain_id,
        block.withdrawals,
        block.header.parent_beacon_block_root,
        calculate_excess_blob_gas(parent_header),
        sender_addresses,
        total_intrinsic_gas,
    )

    chain.last_block_gas_used = apply_body_output.block_gas_used
    chain.last_block_logs_bloom = apply_body_output.block_logs_bloom
    chain.last_receipt_root = apply_body_output.receipt_root
    chain.blocks.append(block)
    if len(chain.blocks) > 255:
        # Real clients have to store more blocks to deal with reorgs, but the
        # protocol only requires the last 255
        chain.blocks = chain.blocks[-255:]


def calculate_base_fee_per_gas(
    block_gas_limit: Uint,
    parent_gas_limit: Uint,
    parent_gas_used: Uint,
    parent_base_fee_per_gas: Uint,
) -> Uint:
    """
    Calculates the base fee per gas for the block.

    Parameters
    ----------
    block_gas_limit :
        Gas limit of the block for which the base fee is being calculated.
    parent_gas_limit :
        Gas limit of the parent block.
    parent_gas_used :
        Gas used in the parent block.
    parent_base_fee_per_gas :
        Base fee per gas of the parent block.

    Returns
    -------
    base_fee_per_gas : `Uint`
        Base fee per gas for the block.
    """
    parent_gas_target = parent_gas_limit // ELASTICITY_MULTIPLIER
    if not check_gas_limit(block_gas_limit, parent_gas_limit):
        raise InvalidBlock

    if parent_gas_used == parent_gas_target:
        expected_base_fee_per_gas = parent_base_fee_per_gas
    elif parent_gas_used > parent_gas_target:
        gas_used_delta = parent_gas_used - parent_gas_target

        parent_fee_gas_delta = parent_base_fee_per_gas * gas_used_delta
        target_fee_gas_delta = parent_fee_gas_delta // parent_gas_target

        base_fee_per_gas_delta = max(
            target_fee_gas_delta // BASE_FEE_MAX_CHANGE_DENOMINATOR,
            Uint(1),
        )

        expected_base_fee_per_gas = (
            parent_base_fee_per_gas + base_fee_per_gas_delta
        )
    else:
        gas_used_delta = parent_gas_target - parent_gas_used

        parent_fee_gas_delta = parent_base_fee_per_gas * gas_used_delta
        target_fee_gas_delta = parent_fee_gas_delta // parent_gas_target

        base_fee_per_gas_delta = (
            target_fee_gas_delta // BASE_FEE_MAX_CHANGE_DENOMINATOR
        )

        expected_base_fee_per_gas = (
            parent_base_fee_per_gas - base_fee_per_gas_delta
        )

    return Uint(expected_base_fee_per_gas)


def validate_header(header: Header, parent_header: Header) -> None:
    """
    Verifies a block header.

    In order to consider a block's header valid, the logic for the
    quantities in the header should match the logic for the block itself.
    For example the header timestamp should be greater than the block's parent
    timestamp because the block was created *after* the parent block.
    Additionally, the block's number should be directly following the parent
    block's number since it is the next block in the sequence.

    Parameters
    ----------
    header :
        Header to check for correctness.
    parent_header :
        Parent Header of the header to check for correctness
    """
    expected_base_fee_per_gas = calculate_base_fee_per_gas(
        header.gas_limit,
        parent_header.gas_limit,
        header.parent_gas_used,
        parent_header.base_fee_per_gas,
    )
    excess_blob_gas = calculate_excess_blob_gas(parent_header)

    if expected_base_fee_per_gas != header.base_fee_per_gas:
        raise InvalidBlock
    if excess_blob_gas != header.excess_blob_gas:
        raise InvalidBlock
    if header.timestamp <= parent_header.timestamp:
        raise InvalidBlock
    if header.number != parent_header.number + Uint(1):
        raise InvalidBlock
    if len(header.extra_data) > 32:
        raise InvalidBlock
    if header.difficulty != 0:
        raise InvalidBlock
    if header.nonce != b"\x00\x00\x00\x00\x00\x00\x00\x00":
        raise InvalidBlock
    if header.ommers_hash != EMPTY_OMMER_HASH:
        raise InvalidBlock

    block_parent_hash = keccak256(rlp.encode(parent_header))
    if header.parent_hash != block_parent_hash:
        raise InvalidBlock


def check_transaction(
    state: State,
    tx: Transaction,
    sender_address: Address,
    coinbase: Address,
    gas_available: Uint,
    chain_id: U64,
    base_fee_per_gas: Uint,
    excess_blob_gas: U64,
) -> Tuple[bool, Address, Uint, Tuple[VersionedHash, ...]]:
    """
    Check if the transaction is includable in the block.

    Parameters
    ----------
    state :
        Current state.
    tx :
        The transaction.
    gas_available :
        The gas remaining in the block.
    chain_id :
        The ID of the current chain.
    base_fee_per_gas :
        The block base fee.
    excess_blob_gas :
        The excess blob gas.

    Returns
    -------
    sender_address :
        The sender of the transaction.
    effective_gas_price :
        The price to charge for gas when the transaction is executed.
    blob_versioned_hashes :
        The blob versioned hashes of the transaction.

    Raises
    ------
    InvalidBlock :
        If the transaction is not includable.
    """
    sender_account = get_account(state, sender_address)
    coinbase_account = get_account(state, coinbase)

    if isinstance(tx, (FeeMarketTransaction, BlobTransaction)):
        if tx.max_fee_per_gas < base_fee_per_gas:
            sender_gas_price = tx.max_fee_per_gas
        else:
            priority_fee_per_gas = min(
                tx.max_priority_fee_per_gas,
                tx.max_fee_per_gas - base_fee_per_gas,
            )
            sender_gas_price = priority_fee_per_gas + base_fee_per_gas
    else:
        sender_gas_price = tx.gas_price

    if isinstance(tx, BlobTransaction):
        blob_versioned_hashes = tx.blob_versioned_hashes
    else:
        blob_versioned_hashes = ()
    
    max_additional_fee = (tx.gas - calculate_intrinsic_cost(tx)) * base_fee_per_gas
    should_execute_tx = (
        tx.gas <= gas_available
        and sender_account.nonce == tx.nonce
        and Uint(coinbase_account.balance) >= max_additional_fee
        and Uint(sender_account.balance) >= Uint(tx.value)
        and sender_account.code == bytearray()
    )

    return should_execute_tx, sender_address, sender_gas_price, blob_versioned_hashes


def make_receipt(
    tx: Transaction,
    error: Optional[Exception],
    cumulative_gas_used: Uint,
    logs: Tuple[Log, ...],
) -> Union[Bytes, Receipt]:
    """
    Make the receipt for a transaction that was executed.

    Parameters
    ----------
    tx :
        The executed transaction.
    error :
        Error in the top level frame of the transaction, if any.
    cumulative_gas_used :
        The total gas used so far in the block after the transaction was
        executed.
    logs :
        The logs produced by the transaction.

    Returns
    -------
    receipt :
        The receipt for the transaction.
    """
    receipt = Receipt(
        succeeded=error is None,
        cumulative_gas_used=cumulative_gas_used,
        bloom=logs_bloom(logs),
        logs=logs,
    )

    if isinstance(tx, AccessListTransaction):
        return b"\x01" + rlp.encode(receipt)
    elif isinstance(tx, FeeMarketTransaction):
        return b"\x02" + rlp.encode(receipt)
    elif isinstance(tx, BlobTransaction):
        return b"\x03" + rlp.encode(receipt)
    else:
        return receipt


def check_transaction_static(
        tx: Transaction,
        chain_id: U64,
        base_fee_per_gas: U64,
        excess_blob_gas: U64,
        ) -> Tuple[Address, Uint, Uint]:

    if not validate_transaction(tx):
        raise InvalidBlock

    if isinstance(tx, (FeeMarketTransaction, BlobTransaction)):
        if tx.max_fee_per_gas < tx.max_priority_fee_per_gas:
            raise InvalidBlock
        if tx.max_fee_per_gas < base_fee_per_gas:
            raise InvalidBlock
    else:
        if tx.gas_price < base_fee_per_gas:
            raise InvalidBlock

    if isinstance(tx, BlobTransaction):
        if not isinstance(tx.to, Address):
            raise InvalidBlock
        if len(tx.blob_versioned_hashes) == 0:
            raise InvalidBlock
        for blob_versioned_hash in tx.blob_versioned_hashes:
            if blob_versioned_hash[0:1] != VERSIONED_HASH_VERSION_KZG:
                raise InvalidBlock

        blob_gas_price = calculate_blob_gas_price(excess_blob_gas)
        if Uint(tx.max_fee_per_blob_gas) < blob_gas_price:
            raise InvalidBlock
            
    sender_address = recover_sender(chain_id, tx)
        
    return sender_address, calculate_intrinsic_cost(tx), calculate_total_blob_gas(tx)

def check_block_static(
    chain: BlockChain,
    block: Block,
) -> Tuple[List[Address], Uint, Uint]:
    total_intrinsic_gas = Uint(0)
    total_blob_gas_used = Uint(0)
    transactions_trie: Trie[
        Bytes, Optional[Union[Bytes, LegacyTransaction]]
    ] = Trie(secured=False, default=None)
    withdrawals_trie: Trie[Bytes, Optional[Union[Bytes, Withdrawal]]] = Trie(
        secured=False, default=None
    )

    if block.header.parent_gas_used != chain.last_block_gas_used:
        raise InvalidBlock
    if block.header.parent_receipt_root != chain.last_receipt_root:
        raise InvalidBlock
    if block.header.parent_bloom != chain.last_block_logs_bloom:
        raise InvalidBlock
    if state_root(chain.state) != block.header.pre_state_root:
        raise InvalidBlock
    
    if block.ommers != ():
        raise InvalidBlock

    sender_addresses = []
    for i, tx in enumerate(map(decode_transaction, block.transactions)):
        sender_address, intrinsic_gas, blob_gas_used = check_transaction_static(
            tx, 
            chain.chain_id,
            block.header.base_fee_per_gas, 
            block.header.excess_blob_gas
            )
        
        sender_addresses.append(sender_address)
        total_intrinsic_gas += intrinsic_gas
        total_blob_gas_used += blob_gas_used

        trie_set(
            transactions_trie, rlp.encode(Uint(i)), encode_transaction(tx)
        )

    if total_intrinsic_gas > block.header.gas_limit:
        raise InvalidBlock
    if total_blob_gas_used > MAX_BLOB_GAS_PER_BLOCK:
        raise InvalidBlock

    blob_gas_price = calculate_blob_gas_price(block.header.excess_blob_gas)
    upfront_cost = (
        total_intrinsic_gas * block.header.base_fee_per_gas
        + total_blob_gas_used * blob_gas_price
    )
    
    coinbase_account = get_account(chain.state, block.header.coinbase)
    if Uint(coinbase_account.balance) < upfront_cost:
        raise InvalidBlock

    coinbase_balance_after_upfront_cost = (
        Uint(coinbase_account.balance) - upfront_cost
    )

    for i, wd in enumerate(block.withdrawals):
        trie_set(withdrawals_trie, rlp.encode(Uint(i)), rlp.encode(wd))

    if block.header.transactions_root != root(transactions_trie):
        raise InvalidBlock
    if block.header.withdrawals_root != root(withdrawals_trie):
        raise InvalidBlock
    if block.header.blob_gas_used != blob_gas_used:
        raise InvalidBlock
    
    return sender_addresses, coinbase_balance_after_upfront_cost, total_intrinsic_gas
    

@dataclass
class ApplyBodyOutput:
    """
    Output from applying the block body to the present state.

    Contains the following:

    block_gas_used : `ethereum.base_types.Uint`
        Gas used for executing all transactions.
    transactions_root : `ethereum.fork_types.Root`
        Trie root of all the transactions in the block.
    receipt_root : `ethereum.fork_types.Root`
        Trie root of all the receipts in the block.
    block_logs_bloom : `Bloom`
        Logs bloom of all the logs included in all the transactions of the
        block.
    state_root : `ethereum.fork_types.Root`
        State root after all transactions have been executed.
    withdrawals_root : `ethereum.fork_types.Root`
        Trie root of all the withdrawals in the block.
    blob_gas_used : `ethereum.base_types.Uint`
        Total blob gas used in the block.
    """

    block_gas_used: Uint
    receipt_root: Root
    block_logs_bloom: Bloom
    state_root: Root


def apply_body(
    state: State,
    block_hashes: List[Hash32],
    coinbase: Address,
    block_number: Uint,
    base_fee_per_gas: Uint,
    block_gas_limit: Uint,
    block_time: U256,
    prev_randao: Bytes32,
    transactions: Tuple[Union[LegacyTransaction, Bytes], ...],
    chain_id: U64,
    withdrawals: Tuple[Withdrawal, ...],
    parent_beacon_block_root: Root,
    excess_blob_gas: U64,
    sender_addresses: List[Address],
    total_intrinsic_gas: Uint,
) -> ApplyBodyOutput:
    """
    Executes a block.

    Many of the contents of a block are stored in data structures called
    tries. There is a transactions trie which is similar to a ledger of the
    transactions stored in the current block. There is also a receipts trie
    which stores the results of executing a transaction, like the post state
    and gas used. This function creates and executes the block that is to be
    added to the chain.

    Parameters
    ----------
    state :
        Current account state.
    block_hashes :
        List of hashes of the previous 256 blocks in the order of
        increasing block number.
    coinbase :
        Address of account which receives block reward and transaction fees.
    block_number :
        Position of the block within the chain.
    base_fee_per_gas :
        Base fee per gas of within the block.
    block_gas_limit :
        Initial amount of gas available for execution in this block.
    block_time :
        Time the block was produced, measured in seconds since the epoch.
    prev_randao :
        The previous randao from the beacon chain.
    transactions :
        Transactions included in the block.
    ommers :
        Headers of ancestor blocks which are not direct parents (formerly
        uncles.)
    chain_id :
        ID of the executing chain.
    withdrawals :
        Withdrawals to be processed in the current block.
    parent_beacon_block_root :
        The root of the beacon block from the parent block.
    excess_blob_gas :
        Excess blob gas calculated from the previous block.

    Returns
    -------
    apply_body_output : `ApplyBodyOutput`
        Output of applying the block body to the state.
    """
    gas_available = block_gas_limit - total_intrinsic_gas
    receipts_trie: Trie[Bytes, Optional[Union[Bytes, Receipt]]] = Trie(
        secured=False, default=None
    )
    block_logs: Tuple[Log, ...] = ()

    beacon_block_roots_contract_code = get_account(
        state, BEACON_ROOTS_ADDRESS
    ).code

    system_tx_message = Message(
        caller=SYSTEM_ADDRESS,
        target=BEACON_ROOTS_ADDRESS,
        gas=SYSTEM_TRANSACTION_GAS,
        value=U256(0),
        data=parent_beacon_block_root,
        code=beacon_block_roots_contract_code,
        depth=Uint(0),
        current_target=BEACON_ROOTS_ADDRESS,
        code_address=BEACON_ROOTS_ADDRESS,
        should_transfer_value=False,
        is_static=False,
        accessed_addresses=set(),
        accessed_storage_keys=set(),
        parent_evm=None,
    )

    system_tx_env = vm.Environment(
        caller=SYSTEM_ADDRESS,
        origin=SYSTEM_ADDRESS,
        block_hashes=block_hashes,
        coinbase=coinbase,
        number=block_number,
        gas_limit=block_gas_limit,
        base_fee_per_gas=base_fee_per_gas,
        gas_price=base_fee_per_gas,
        time=block_time,
        prev_randao=prev_randao,
        state=state,
        chain_id=chain_id,
        traces=[],
        excess_blob_gas=excess_blob_gas,
        blob_versioned_hashes=(),
        transient_storage=TransientStorage(),
    )

    system_tx_output = process_message_call(system_tx_message, system_tx_env)

    destroy_touched_empty_accounts(
        system_tx_env.state, system_tx_output.touched_accounts
    )

    for i, tx in enumerate(map(decode_transaction, transactions)):
        sender_address = sender_addresses[i]
        intrinsic_gas_cost = calculate_intrinsic_cost(tx)
        gas_available += intrinsic_gas_cost
        (
            should_execute_tx,
            sender_gas_price,
            blob_versioned_hashes,
        ) = check_transaction(
            state,
            tx,
            sender_address,
            gas_available,
            chain_id,
            base_fee_per_gas,
            excess_blob_gas,
        )

        if should_execute_tx:
            env = vm.Environment(
                caller=sender_address,
                origin=sender_address,
                block_hashes=block_hashes,
                coinbase=coinbase,
                number=block_number,
                gas_limit=block_gas_limit,
                base_fee_per_gas=base_fee_per_gas,
                gas_price=sender_gas_price,
                time=block_time,
                prev_randao=prev_randao,
                state=state,
                chain_id=chain_id,
                traces=[],
                excess_blob_gas=excess_blob_gas,
                blob_versioned_hashes=blob_versioned_hashes,
                transient_storage=TransientStorage(),
            )

            gas_used, logs, error = process_transaction(env, tx)
            gas_available -= gas_used

            receipt = make_receipt(
                tx, error, (block_gas_limit - gas_available), logs
            )

            trie_set(
                receipts_trie,
                rlp.encode(Uint(i)),
                receipt,
            )

            block_logs += logs
        else:
            gas_available -= intrinsic_gas_cost

    block_gas_used = block_gas_limit - gas_available
    block_logs_bloom = logs_bloom(block_logs)

    for i, wd in enumerate(withdrawals):
        process_withdrawal(state, wd)

        if account_exists_and_is_empty(state, wd.address):
            destroy_account(state, wd.address)

    return ApplyBodyOutput(
        block_gas_used,
        root(receipts_trie),
        block_logs_bloom,
        state_root(state),
    )


def process_transaction(
    env: vm.Environment, tx: Transaction
) -> Tuple[Uint, Tuple[Log, ...], Optional[Exception]]:
    """
    Execute a transaction against the provided environment.

    This function processes the actions needed to execute a transaction.
    It decrements the sender's account after calculating the gas fee and
    refunds them the proper amount after execution. Calling contracts,
    deploying code, and incrementing nonces are all examples of actions that
    happen within this function or from a call made within this function.

    Accounts that are marked for deletion are processed and destroyed after
    execution.

    Parameters
    ----------
    env :
        Environment for the Ethereum Virtual Machine.
    tx :
        Transaction to execute.

    Returns
    -------
    gas_left : `ethereum.base_types.U256`
        Remaining gas after execution.
    logs : `Tuple[ethereum.blocks.Log, ...]`
        Logs generated during execution.
    """
    sender = env.origin
    sender_account = get_account(env.state, sender)
    coinbase_account = get_account(env.state, env.coinbase)

    intrinsic_gas_cost = calculate_intrinsic_cost(tx)
    gas = tx.gas - intrinsic_gas_cost
    gas_fee = gas * env.base_fee_per_gas
    increment_nonce(env.state, sender)

    coinbase_balance_after_gas_fee = (
        Uint(coinbase_account.balance) - gas_fee
    )
    set_account_balance(env.state, env.coinbase, U256(coinbase_balance_after_gas_fee))

    preaccessed_addresses = set()
    preaccessed_storage_keys = set()
    preaccessed_addresses.add(env.coinbase)
    if isinstance(
        tx, (AccessListTransaction, FeeMarketTransaction, BlobTransaction)
    ):
        for address, keys in tx.access_list:
            preaccessed_addresses.add(address)
            for key in keys:
                preaccessed_storage_keys.add((address, key))

    message = prepare_message(
        sender,
        tx.to,
        tx.value,
        tx.data,
        gas,
        env,
        preaccessed_addresses=frozenset(preaccessed_addresses),
        preaccessed_storage_keys=frozenset(preaccessed_storage_keys),
    )

    output = process_message_call(message, env)

    gas_used = tx.gas - output.gas_left
    gas_refund = min(gas_used // Uint(5), Uint(output.refund_counter))
    gas_refund_amount = (output.gas_left + gas_refund) * env.base_fee_per_gas

    total_gas_used = gas_used - gas_refund
    sender_fee = min(sender_account.balance, U256(total_gas_used * env.gas_price))

    sender_balance_after_fees = sender_account.balance - sender_fee
    set_account_balance(
        env.state, 
        sender, 
        sender_balance_after_fees,
    )

    # refund gas and transfer miner fees
    coinbase_balance_after_refund_and_fees = (
        coinbase_account.balance 
        + U256(gas_refund_amount)
        + sender_fee
    )
    if coinbase_balance_after_refund_and_fees != 0:
        set_account_balance(
            env.state, 
            env.coinbase, 
            coinbase_balance_after_refund_and_fees,
        )
    elif account_exists_and_is_empty(env.state, env.coinbase):
        destroy_account(env.state, env.coinbase)

    for address in output.accounts_to_delete:
        destroy_account(env.state, address)

    destroy_touched_empty_accounts(env.state, output.touched_accounts)

    return total_gas_used, output.logs, output.error


def compute_header_hash(header: Header) -> Hash32:
    """
    Computes the hash of a block header.

    The header hash of a block is the canonical hash that is used to refer
    to a specific block and completely distinguishes a block from another.

    ``keccak256`` is a function that produces a 256 bit hash of any input.
    It also takes in any number of bytes as an input and produces a single
    hash for them. A hash is a completely unique output for a single input.
    So an input corresponds to one unique hash that can be used to identify
    the input exactly.

    Prior to using the ``keccak256`` hash function, the header must be
    encoded using the Recursive-Length Prefix. See :ref:`rlp`.
    RLP encoding the header converts it into a space-efficient format that
    allows for easy transfer of data between nodes. The purpose of RLP is to
    encode arbitrarily nested arrays of binary data, and RLP is the primary
    encoding method used to serialize objects in Ethereum's execution layer.
    The only purpose of RLP is to encode structure; encoding specific data
    types (e.g. strings, floats) is left up to higher-order protocols.

    Parameters
    ----------
    header :
        Header of interest.

    Returns
    -------
    hash : `ethereum.crypto.hash.Hash32`
        Hash of the header.
    """
    return keccak256(rlp.encode(header))


def check_gas_limit(gas_limit: Uint, parent_gas_limit: Uint) -> bool:
    """
    Validates the gas limit for a block.

    The bounds of the gas limit, ``max_adjustment_delta``, is set as the
    quotient of the parent block's gas limit and the
    ``GAS_LIMIT_ADJUSTMENT_FACTOR``. Therefore, if the gas limit that is
    passed through as a parameter is greater than or equal to the *sum* of
    the parent's gas and the adjustment delta then the limit for gas is too
    high and fails this function's check. Similarly, if the limit is less
    than or equal to the *difference* of the parent's gas and the adjustment
    delta *or* the predefined ``GAS_LIMIT_MINIMUM`` then this function's
    check fails because the gas limit doesn't allow for a sufficient or
    reasonable amount of gas to be used on a block.

    Parameters
    ----------
    gas_limit :
        Gas limit to validate.

    parent_gas_limit :
        Gas limit of the parent block.

    Returns
    -------
    check : `bool`
        True if gas limit constraints are satisfied, False otherwise.
    """
    max_adjustment_delta = parent_gas_limit // GAS_LIMIT_ADJUSTMENT_FACTOR
    if gas_limit >= parent_gas_limit + max_adjustment_delta:
        return False
    if gas_limit <= parent_gas_limit - max_adjustment_delta:
        return False
    if gas_limit < GAS_LIMIT_MINIMUM:
        return False

    return True
