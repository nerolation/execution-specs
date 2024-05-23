from functools import partial
from typing import Dict, Generator, Tuple

import pytest

from ethereum import rlp
from ethereum.base_types import U256, Bytes, Bytes8, Bytes32, Uint
from ethereum.crypto.hash import Hash32
from ethereum.exceptions import InvalidBlock, RLPDecodingError
from tests.helpers import TEST_FIXTURES
from tests.helpers.load_state_tests import (
    Load,
    fetch_state_test_files,
    idfn,
    run_blockchain_st_test,
)

fetch_prague_tests = partial(fetch_state_test_files, network="Prague")

FIXTURES_LOADER = Load("Prague", "prague")

run_prague_blockchain_st_tests = partial(
    run_blockchain_st_test, load=FIXTURES_LOADER
)

ETHEREUM_TESTS_PATH = TEST_FIXTURES["ethereum_tests"]["fixture_path"]
ETHEREUM_SPEC_TESTS_PATH = TEST_FIXTURES["execution_spec_tests"][
    "fixture_path"
]


# Run state tests
test_dir = f"{ETHEREUM_TESTS_PATH}/BlockchainTests/"

SLOW_TESTS = (
    # GeneralStateTests
    "stTimeConsuming/CALLBlake2f_MaxRounds.json",
    "stTimeConsuming/static_Call50000_sha256.json",
    "vmPerformance/loopExp.json",
    "vmPerformance/loopMul.json",
    "QuadraticComplexitySolidity_CallDataCopy_d0g1v0_Prague",
    "CALLBlake2f_d9g0v0_Prague",
    "CALLCODEBlake2f_d9g0v0",
    # GeneralStateTests
    "stRandom/randomStatetest177.json",
    "stCreateTest/CreateOOGafterMaxCodesize.json",
    # ValidBlockTest
    "bcExploitTest/DelegateCallSpam.json",
    # InvalidBlockTest
    "bcUncleHeaderValidity/nonceWrong.json",
    "bcUncleHeaderValidity/wrongMixHash.json",
)

# These are tests that are considered to be incorrect,
# Please provide an explanation when adding entries
IGNORE_TESTS = (
    # ValidBlockTest
    "bcForkStressTest/ForkStressTest.json",
    "bcGasPricerTest/RPC_API_Test.json",
    "bcMultiChainTest",
    "bcTotalDifficultyTest",
    # InvalidBlockTest
    "bcForgedTest",
    "bcMultiChainTest",
    "GasLimitHigherThan2p63m1_Prague",
    # TODO: The below tests are being ignored due to a bug in
    # upstream repo. They should be removed from the ignore list
    # once the bug is resolved
    # See: https://github.com/ethereum/execution-spec-tests/pull/134
    "Pyspecs/vm/chain_id.json",
    "Pyspecs/vm/dup.json",
    "Pyspecs/example/yul.json",
    "Pyspecs/eips/warm_coinbase_gas_usage.json",
    "Pyspecs/eips/warm_coinbase_call_out_of_gas.json",
)

# All tests that recursively create a large number of frames (50000)
BIG_MEMORY_TESTS = (
    # GeneralStateTests
    "50000_",
    "/stQuadraticComplexityTest/",
    "/stRandom2/",
    "/stRandom/",
    "/stSpecialTest/",
    "stTimeConsuming/",
    "stBadOpcode/",
    "stStaticCall/",
)

fetch_state_tests = partial(
    fetch_prague_tests,
    test_dir,
    ignore_list=IGNORE_TESTS,
    slow_list=SLOW_TESTS,
    big_memory_list=BIG_MEMORY_TESTS,
)


@pytest.mark.parametrize(
    "test_case",
    fetch_state_tests(),
    ids=idfn,
)
def test_general_state_tests(test_case: Dict) -> None:
    run_prague_blockchain_st_tests(test_case)


# Run temporary test fixtures for Prague
test_dirs = (
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip7002_el_triggerable_withdrawals",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip6110_deposits/deposits",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_g1add",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_g1mul",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_g2add",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_g2mul",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_pairing",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_g1msm",
    "tests/fixtures/latest_fork_tests/fixtures/blockchain_tests/prague/eip2537_bls_12_381_precompiles/bls12_g2msm",
)


def fetch_temporary_tests(test_dirs: Tuple[str, ...]) -> Generator:
    """
    Fetch the relevant tests for a particular EIP-Implementation
    from among the temporary fixtures from ethereum-spec-tests.
    """
    for test_dir in test_dirs:
        yield from fetch_prague_tests(test_dir)


@pytest.mark.parametrize(
    "test_case",
    fetch_temporary_tests(test_dirs),
    ids=idfn,
)
def test_execution_specs_generated_tests(test_case: Dict) -> None:
    run_prague_blockchain_st_tests(test_case)
