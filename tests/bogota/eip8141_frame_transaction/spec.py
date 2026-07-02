"""Reference spec for [EIP-8141: Frame Transaction.](https://eips.ethereum.org/EIPS/eip-8141)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSpec:
    """Reference specification."""

    git_path: str
    version: str


ref_spec_8141 = ReferenceSpec(
    git_path="EIPS/eip-8141.md",
    version="7e54d3de551b268eee131fa795549fb7f6850bf4",
)


class Spec:
    """Constants and parameters from EIP-8141."""

    # Transaction type and top-level costs
    FRAME_TX_TYPE: int = 0x06
    FRAME_TX_INTRINSIC_COST: int = 15_000
    FRAME_TX_PER_FRAME_COST: int = 475

    # Protocol addresses
    ENTRY_POINT: int = 0xAA
    EXPIRY_VERIFIER: int = 0x8141

    # Expiry verifier parameters
    EXPIRY_DATA_LENGTH: int = 8
    # Runtime code installed at EXPIRY_VERIFIER at activation.
    EXPIRY_VERIFIER_CODE: bytes = bytes.fromhex(
        "60083614600a575f5ffd5b5f3560c01c4211601657005b5f5ffd"
    )

    # Structural limits
    MAX_FRAMES: int = 64

    # Frame modes
    MODE_DEFAULT: int = 0
    MODE_VERIFY: int = 1
    MODE_SENDER: int = 2

    # Frame flags (bit positions, zero-based)
    APPROVE_SCOPE_MASK: int = 0x3  # bits 0-1
    ATOMIC_BATCH_FLAG: int = 0x4  # bit 2

    # APPROVE scope operand bitmask
    APPROVE_NONE: int = 0x0
    APPROVE_PAYMENT: int = 0x1
    APPROVE_EXECUTION: int = 0x2
    APPROVE_EXECUTION_AND_PAYMENT: int = 0x3

    # Signature schemes and their verification gas costs
    SCHEME_SECP256K1: int = 0x0
    SCHEME_P256: int = 0x1
    SIG_GAS_SECP256K1: int = 2_800
    SIG_GAS_P256: int = 6_700

    # New instructions
    APPROVE: int = 0xAA
    TXPARAM: int = 0xB0
    FRAMEDATALOAD: int = 0xB1
    FRAMEDATACOPY: int = 0xB2
    FRAMEPARAM: int = 0xB3
    SIGPARAM: int = 0xB4

    # Receipt status codes
    STATUS_FAILURE: int = 0x0
    STATUS_SUCCESS: int = 0x1
    STATUS_SKIPPED: int = 0x3  # skipped due to failed atomic batch

    # Mempool parameters
    MAX_VERIFY_GAS: int = 100_000
    MAX_PENDING_TXS_USING_NON_CANONICAL_PAYMASTER: int = 1

    # Numeric field limits (from static constraints)
    MAX_NONCE: int = 2**64 - 1
    MAX_GAS_LIMIT: int = 2**64 - 1
