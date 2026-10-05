"""
EIP-8131: Unified Transaction Content Floor.

Price every transaction content byte -- calldata, access list entries,
EIP-7702 authorizations, and blob versioned hashes -- into the floor at
one per-byte rate.

https://eips.ethereum.org/EIPS/eip-8131
"""

from typing import List, Sized

from execution_testing.base_types import AccessList
from execution_testing.base_types.conversions import BytesConvertible

from .....recipient_type import RecipientType
from ....base_fork import BaseFork, TransactionDataFloorCostCalculator
from ...helpers import count_or_len

AUTH_TUPLE_BYTES = 108
BLOB_VERSIONED_HASH_BYTES = 32


class EIP8131(BaseFork):
    """EIP-8131 class."""

    @classmethod
    def transaction_data_floor_cost_calculator(
        cls,
    ) -> TransactionDataFloorCostCalculator:
        """
        Add the authorization tuples and blob versioned hashes to the
        inherited floor, which already prices calldata and access list
        bytes at the same rate.
        """
        super_fn = super(EIP8131, cls).transaction_data_floor_cost_calculator()
        gas_costs = cls.gas_costs()

        def fn(
            *,
            data: BytesConvertible,
            access_list: List[AccessList] | None = None,
            contract_creation: bool = False,
            sends_value: bool = False,
            recipient_type: RecipientType = RecipientType.CONTRACT,
            authorization_list_or_count: Sized | int | None = None,
            blob_versioned_hashes_or_count: Sized | int | None = None,
        ) -> int:
            content_bytes = (
                count_or_len(authorization_list_or_count) * AUTH_TUPLE_BYTES
                + count_or_len(blob_versioned_hashes_or_count)
                * BLOB_VERSIONED_HASH_BYTES
            )
            return (
                super_fn(
                    data=data,
                    access_list=access_list,
                    contract_creation=contract_creation,
                    sends_value=sends_value,
                    recipient_type=recipient_type,
                    authorization_list_or_count=authorization_list_or_count,
                    blob_versioned_hashes_or_count=blob_versioned_hashes_or_count,
                )
                + content_bytes * gas_costs.FLOOR_GAS_PER_BYTE
            )

        return fn
