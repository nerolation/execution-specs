"""Reference spec for [EIP-8279: Block Access List Byte Floor](https://eips.ethereum.org/EIPS/eip-8279)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSpec:
    """Reference specification."""

    git_path: str
    version: str


ref_spec_8279 = ReferenceSpec(
    git_path="EIPS/eip-8279.md",
    version="f6b71f0e7e554dbf6864c2c153438bce69300125",
)
