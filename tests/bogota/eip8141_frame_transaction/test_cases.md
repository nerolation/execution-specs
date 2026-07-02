# EIP-8141 Frame Transaction Test Cases

Test plan for [EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141),
spec version `7e54d3de551b268eee131fa795549fb7f6850bf4`.

EIP-8141 adds a type-`0x06` transaction that splits execution into an ordered list of
*frames* (validate, approve payment, execute), plus the `APPROVE` (`0xaa`) and introspection
(`0xb0`–`0xb4`) instructions, account "default code", and an expiry-verifier predeploy at
`0x8141`.

Sections 1–13 cover consensus behavior (`📝 Planned`); section 14 lists the deferred,
non-consensus mempool policy (`⏭️ Deferred`).

---

## 1. Happy-path execution (default code)

EOA sender using the built-in "default code" — no smart-account contract required.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_frame_tx_self_relay_simple` | A minimal `[self_verify, user_op]` frame tx executes and self-pays via default code (EIP Example 1). | EOA `sender` (default code, funded). Frame 0: `VERIFY`, target=null(sender), flags=`APPROVE_EXECUTION_AND_PAYMENT`, empty data; one `SECP256K1` signature with empty `msg` over `compute_sig_hash(tx)`. Frame 1: `SENDER`, target=`Bob`, value=0, flags=`APPROVE_NONE`, calldata. | Both frames succeed; `payer == sender`; `sender_approved` set by frame 0; sender nonce incremented exactly once (at payment approval). Receipt records `payer=sender` and two frame receipts, each `status=1`. | 📝 Planned |
| `test_frame_tx_simple_eth_transfer` | A `SENDER` frame performs a native value transfer (EIP Example 1a). | Frame 0: default-code `self_verify` (as above). Frame 1: `SENDER`, target=`Bob`, `value=amount`, empty data. | Frame 1 moves `amount` wei sender→`Bob` with ordinary CALL value semantics; `Bob += amount`; `sender` debited `amount` + gas. | 📝 Planned |
| `test_frame_tx_receipt_encoding` | The receipt encodes `[cumulative_gas_used, payer, [frame_receipt, ...]]` with `frame_receipt = [status, gas_used, logs]`. | Multi-frame tx where a `SENDER` frame emits a `LOG`. | `payer` equals the resolved target that approved payment; each frame contributes one `frame_receipt`; the log appears under its frame's receipt; `cumulative_gas_used` matches total gas used. | 📝 Planned |
| `test_frame_tx_multiple_sender_frames` | Multiple non-atomic `SENDER` frames execute independently and in order. | Frame 0 `self_verify`; frames 1..N `SENDER` targeting different contracts/values, none with `ATOMIC_BATCH_FLAG`. | All frames execute in order, each with its own `gas_limit`; a revert in one frame does not roll back the others. | 📝 Planned |
| `test_user_op_revert_does_not_invalidate_tx` | A post-payment (non-atomic) frame revert discards only that frame's state; the tx stays valid. | After payment is approved, a `SENDER` frame reverts; a later `SENDER` frame succeeds. | Reverted frame `status=0`, its state discarded but gas still consumed; the later frame applies; tx valid; payer charged. | 📝 Planned |

---

## 2. Static validity constraints

Each row is one malformation of an otherwise-valid `self_verify` tx; all are rejected pre-execution.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_frame_tx_empty_frames_invalid` | `len(frames) == 0` is invalid. | `frames = []`. | Invalid (`assert len(tx.frames) > 0`). | 📝 Planned |
| `test_frame_tx_wrong_chain_id_invalid` | `chain_id` must match the current chain (replay protection). | `chain_id` set to a value other than the node's chain. | Invalid. | 📝 Planned |
| `test_frame_tx_frame_count_boundary` | `len(frames) <= MAX_FRAMES` (=64). | 64 frames (valid) vs 65 frames (invalid). | 64 valid; 65 invalid. | 📝 Planned |
| `test_frame_tx_nonce_too_large_invalid` | `nonce >= 2**64` is invalid. | `nonce = 2**64`. | Invalid (`assert tx.nonce < 2**64`). | 📝 Planned |
| `test_frame_tx_sender_wrong_length_invalid` | `len(sender) != 20` is invalid. | 19-/21-byte sender. | Invalid. | 📝 Planned |
| `test_frame_tx_nonce_mismatch_invalid` | `tx.nonce != state[sender].nonce` is invalid. | `nonce` one off the account nonce. | Invalid (first processing step). | 📝 Planned |
| `test_frame_tx_invalid_mode_invalid` | `frame.mode >= 3` is invalid. | A frame with `mode = 3`. | Invalid (`assert frame.mode < 3`). | 📝 Planned |
| `test_frame_tx_invalid_flags_invalid` | `frame.flags >= 8` is invalid. | A frame with `flags = 8`. | Invalid (`assert frame.flags < 8`). | 📝 Planned |
| `test_frame_tx_target_wrong_length_invalid` | `frame.target` is neither null nor 20 bytes. | 19-byte target. | Invalid. Companion: null target is valid (resolves to `sender`). | 📝 Planned |
| `test_frame_tx_nonzero_value_non_sender_invalid` | Only `SENDER` frames may carry non-zero `value`. | `VERIFY`/`DEFAULT` frame with `value = 1`. | Invalid (`assert frame.mode == SENDER or frame.value == 0`). | 📝 Planned |
| `test_frame_tx_gas_limit_overflow_invalid` | Per-frame and cumulative `gas_limit` bounds. | (a) one frame `gas_limit = 2**64`; (b) frames whose sum exceeds `2**64 - 1`. | Both invalid. | 📝 Planned |
| `test_frame_tx_atomic_flag_on_last_frame_invalid` | `ATOMIC_BATCH_FLAG` on the last frame is invalid. | Final frame has bit 2 set. | Invalid (`assert i + 1 < len(tx.frames)`). | 📝 Planned |
| `test_frame_tx_blob_fee_without_blobs_invalid` | `max_fee_per_blob_gas` must be 0 when `blob_versioned_hashes` is empty. | Empty blob list, non-zero `max_fee_per_blob_gas`. | Invalid. | 📝 Planned |

---

## 3. Signature list validation

All signatures are validated before any frame runs; any invalid one voids the tx.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_sig_secp256k1_empty_msg_valid` | `SECP256K1` signature over the canonical sig hash validates. | One sig: `scheme=SECP256K1`, `msg=b""`, correct `signer`, valid 65-byte signature over `compute_sig_hash(tx)`. | Signature validates; default-code `self_verify` approves. | 📝 Planned |
| `test_sig_secp256k1_explicit_msg_valid` | A signature over an explicit non-zero 32-byte `msg` validates and is committed (not elided). | Sender `self_verify` signature (empty `msg`) **plus** a second `SECP256K1` signature over an explicit non-zero 32-byte digest. | Both validate; the explicit-`msg` entry's raw bytes are committed by `compute_sig_hash` (not elided). | 📝 Planned |
| `test_sig_zero_explicit_msg_invalid` | Explicit 32-byte all-zero `msg` is reserved and invalid. | `msg = b"\x00" * 32`. | Invalid. | 📝 Planned |
| `test_sig_bad_msg_length_invalid` | `len(msg)` other than 0 or 32 is invalid. | `len(msg) = 16`. | Invalid. | 📝 Planned |
| `test_sig_wrong_signer_invalid` | `signer != ecrecover(...)` fails. | Valid signature but `signer` set to a different address. | Invalid. | 📝 Planned |
| `test_sig_bad_length_invalid` | Wrong raw signature length for the scheme fails. | `SECP256K1` with `len != 65`; `P256` with `len != 128`. | Invalid. | 📝 Planned |
| `test_sig_unknown_scheme_invalid` | `scheme` in `0x2..0xff` is invalid. | `scheme = 2`. | Invalid. | 📝 Planned |
| `test_sig_signer_wrong_length_invalid` | For `SECP256K1`/`P256`, `len(signer) != 20` is invalid. | 19-byte signer. | Invalid (`assert len(sig.signer) == 20`). | 📝 Planned |
| `test_sig_p256_valid` | `P256` signature validates and derives the signer correctly. | `scheme=P256`, 128-byte `r\|\|s\|\|qx\|\|qy`, `signer == keccak256(qx\|\|qy)[12:]`. | Signature validates. | 📝 Planned |
| `test_sig_p256_signer_mismatch_invalid` | `P256` with `signer != keccak256(qx\|\|qy)[12:]` fails. | Valid curve point, mismatched signer address. | Invalid. | 📝 Planned |
| `test_sig_hash_elides_empty_msg_bytes` | Raw bytes of empty-`msg` signatures are elided from `compute_sig_hash`. | Two txs identical except the raw signature bytes of an empty-`msg` entry. | `compute_sig_hash` is identical (empty-`msg` bytes elided). | 📝 Planned |
| `test_sig_all_must_pass` | Every signature must validate, even unused ones. | Two signatures; the second (never referenced by a frame) is corrupted. | Invalid (all signatures validated up front). | 📝 Planned |
| `test_frame_tx_empty_signatures_list` | The `signatures` list is optional; a smart account may self-verify without it. | Smart-account `sender` whose `VERIFY` code calls `APPROVE` via its own logic; `tx.signatures = []`. | Transaction valid; verification and approval succeed with no signature entries. | 📝 Planned |

---

## 4. Modes, resolved target, and default code

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_mode_default_caller_is_entry_point` | `DEFAULT` frame runs with `caller = ENTRY_POINT`. | `DEFAULT` frame to a contract that stores `CALLER`. | Stored caller == `ENTRY_POINT` (`0xaa`). | 📝 Planned |
| `test_mode_verify_caller_is_entry_point` | `VERIFY` frame runs with `caller = ENTRY_POINT`. | `VERIFY` frame to a smart-account contract that records `CALLER`. | Recorded caller == `ENTRY_POINT`. | 📝 Planned |
| `test_mode_sender_caller_is_sender` | `SENDER` frame runs with `caller = tx.sender`. | Approved sender; `SENDER` frame to a contract storing `CALLER`. | Stored caller == `tx.sender`. | 📝 Planned |
| `test_mode_sender_requires_approval` | A `SENDER` frame before `sender_approved` is invalid. | `SENDER` frame ordered before any approving `VERIFY` frame. | Invalid (`sender_approved` must be true). | 📝 Planned |
| `test_resolved_target_null_is_sender` | Null `target` resolves to `tx.sender`. | Frame with `target = None`. | Execution targets `tx.sender`'s (default) code; `resolved_target == sender`. | 📝 Planned |
| `test_verify_is_staticcall` | `VERIFY` frames execute as `STATICCALL` (state writes banned except `APPROVE`). | `VERIFY` frame to a contract that attempts `SSTORE`. | Frame reverts → transaction invalid. | 📝 Planned |
| `test_default_code_verify_approves` | Default-code `VERIFY` approves the allowed scope with a matching sender signature. | EOA sender (default code); `VERIFY` frame flags carry an allowed scope; matching empty-`msg` `SECP256K1` signature present. | Default code calls `APPROVE(allowed_scope)`; approval recorded. | 📝 Planned |
| `test_default_code_verify_no_scope_reverts` | Default-code `VERIFY` with `allowed_scope == APPROVE_NONE` reverts. | `VERIFY` frame flags with no approval-scope bits. | Frame reverts → transaction invalid. | 📝 Planned |
| `test_default_code_verify_execution_scope_requires_self` | Default-code `VERIFY` with `APPROVE_EXECUTION` requires `resolved_target == tx.sender`. | Execution-scope flag but `target != sender`. | Frame reverts. | 📝 Planned |
| `test_default_code_verify_missing_signature_reverts` | Default-code `VERIFY` reverts without a matching sender `SECP256K1`/empty-`msg` signature. | No signature with `signer == resolved_target` and `msg == b""`. | Frame reverts → transaction invalid. | 📝 Planned |
| `test_default_code_sender_returns_empty` | Default-code `SENDER`/`DEFAULT` frames return successfully like empty code. | `SENDER` frame targeting an EOA with default code, empty data. | Frame succeeds with no state effect beyond value transfer. | 📝 Planned |
| `test_smart_account_delegated_7702` | A `resolved_target` with an EIP-7702 delegation indicator executes the delegated code. | Sender delegated (7702) to a smart-account implementation; `VERIFY` frame. | Delegated code runs and may call `APPROVE`. | 📝 Planned |

---

## 5. `APPROVE` instruction (`0xaa`)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_approve_payment_sets_payer` | `APPROVE(APPROVE_PAYMENT)` sets `payer`, increments the sender nonce, and collects max cost. | `[only_verify, pay]` prefix where `pay` approves payment. | `payer` = the approving resolved target; sender nonce++; max cost collected from `payer`; end-of-tx refund returns unused gas to `payer`. | 📝 Planned |
| `test_approve_execution_sets_sender_approved` | `APPROVE(APPROVE_EXECUTION)` sets `sender_approved` (only when `resolved_target == sender`). | `only_verify` frame targeting sender approves execution. | `sender_approved = true`; subsequent `SENDER` frames permitted. | 📝 Planned |
| `test_approve_execution_and_payment` | `APPROVE(APPROVE_EXECUTION_AND_PAYMENT)` sets both in one call. | `self_verify` frame targeting sender, flags `=0x3`. | Both `sender_approved` and `payer` set; sender nonce++; max cost collected. | 📝 Planned |
| `test_approve_wrong_address_reverts` | `APPROVE` reverts when executed where `ADDRESS != resolved_target`. | Resolved target makes a plain `CALL` into a helper; the helper (whose `ADDRESS` is itself, not the resolved target) executes `APPROVE`. | Frame reverts. | 📝 Planned |
| `test_approve_via_delegatecall_succeeds` | `APPROVE` succeeds from a library `DELEGATECALL`ed by the resolved target (`ADDRESS` preserved). | Resolved target `DELEGATECALL`s a lib that calls `APPROVE`. | Approval recorded. | 📝 Planned |
| `test_approve_terminates_frame_like_return` | `APPROVE` exits the frame successfully like `RETURN`. | Frame runs `APPROVE(scope)` followed by more opcodes. | Frame exits at `APPROVE`; trailing opcodes do not run; approval applied. | 📝 Planned |
| `test_approve_scope_not_in_flags_reverts` | `APPROVE(scope)` reverts when `scope` is not permitted by `frame.flags`. | Flags allow only `APPROVE_EXECUTION`; code calls `APPROVE(APPROVE_PAYMENT)`. | Frame reverts (`scope & ~(flags & APPROVE_SCOPE_MASK) != 0`). | 📝 Planned |
| `test_approve_zero_scope_reverts` | `APPROVE(0)` reverts. | Code calls `APPROVE(APPROVE_NONE)`. | Frame reverts (`scope != 0` required). | 📝 Planned |
| `test_approve_execution_not_sender_reverts` | `APPROVE_EXECUTION` reverts when `resolved_target != tx.sender`. | Non-sender target approves execution. | Frame reverts. | 📝 Planned |
| `test_approve_execution_already_approved_reverts` | A second `APPROVE_EXECUTION` reverts once `sender_approved` is set. | Two frames both approving execution. | Second frame reverts. | 📝 Planned |
| `test_approve_payment_already_set_reverts` | A second `APPROVE_PAYMENT` reverts once `payer` is set. | Two `pay` frames. | Second reverts. | 📝 Planned |
| `test_approve_payment_insufficient_balance_reverts` | `APPROVE_PAYMENT` reverts if `resolved_target` cannot cover max cost. | Payer under-funded relative to `TXPARAM(0x06)`. | Frame reverts → transaction invalid. | 📝 Planned |
| `test_approve_payment_requires_sender_approved` | `APPROVE_PAYMENT` reverts if `sender_approved == false`. | `pay` frame before sender approval. | Frame reverts. | 📝 Planned |
| `test_payer_never_set_invalid` | The transaction is invalid if no frame ever sets `payer`. | `[only_verify, user_op]`: `VERIFY` frame calls `APPROVE(APPROVE_EXECUTION)` only, so `sender_approved` is set but payment is never approved. | Invalid at the post-loop `payer != None` check. | 📝 Planned |

---

## 6. Gas accounting

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_gas_intrinsic_and_per_frame_cost` | The fixed overhead equals `FRAME_TX_INTRINSIC_COST` + `len(frames)*FRAME_TX_PER_FRAME_COST` + `calldata_cost(rlp(frames))` + `calldata_cost(rlp(signatures))` + signature verification cost. | Vary frame count and signature count; measure the gas required beyond `sum(frame.gas_limit)`. | Fixed overhead matches the formula; `tx_gas_limit` = overhead + `sum(frame.gas_limit)`. | 📝 Planned |
| `test_gas_signature_verification_cost` | Signature verification costs `2800`/`6700` per `SECP256K1`/`P256` signature. | Txs with 1 secp, 1 p256, and mixed. | Charged gas increases by the exact per-scheme amount. | 📝 Planned |
| `test_gas_calldata_cost_eip7623` | Calldata cost of `rlp(frames)` and `rlp(signatures)` follows EIP-7623 (scheme-dependent signer bytes included). | Vary zero/non-zero bytes in frames/signatures. | Charged calldata cost matches EIP-7623 over the encoded objects. | 📝 Planned |
| `test_gas_per_frame_isolation` | Unused gas from a frame is NOT available to later frames. | Frame 1 with a generous limit that finishes early; frame 2 with a tight limit that would pass only if it inherited leftover. | Frame 2 OOGs at its own limit; no carry-over. | 📝 Planned |
| `test_gas_refund_to_payer` | `refund = sum(frame.gas_limit) - total_gas_used` returned to payer and added to the block gas pool. | Frames that under-use their limits. | Payer credited the unused portion at `effective_gas_price`; block gas pool restored. | 📝 Planned |
| `test_gas_max_cost_txparam` | `TXPARAM(0x06)` returns the max cost (basefee=max, all gas used, incl. blob + intrinsic + signature costs) that is actually collected. | Tx with blobs and multiple signatures. | The value read equals the amount debited from payer at payment approval. | 📝 Planned |
| `test_gas_sender_value_separate_from_fee` | `frame.value` transfers are separate from `tx_fee` and use ordinary CALL value semantics/costs. | `SENDER` frame with non-zero value. | Value moves sender→target at ordinary CALL cost; fee accounting unaffected. | 📝 Planned |
| `test_sender_value_insufficient_balance_reverts` | A frame whose `value` exceeds the caller's balance reverts (ordinary CALL semantics). | Non-atomic `SENDER` frame with `value` > `sender` balance. | Frame reverts (`status=0`), state discarded; tx remains valid; other frames unaffected. | 📝 Planned |

---

## 7. Atomic batching (`ATOMIC_BATCH_FLAG`, bit 2)

An **atomic batch** is a maximal contiguous run `[i, j]` where frames `i..j-1` have the flag
set and frame `j` does not.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_atomic_batch_success` | A fully-successful atomic batch applies all its frames (EIP Example 2: approve + swap). | `self_verify`, then frame A (`ATOMIC_BATCH_FLAG`) + frame B (no flag). | Both A and B applied; receipts `status=1`. | 📝 Planned |
| `test_atomic_batch_revert_unrolls` | A failure inside a batch rolls the whole batch back to its pre-batch state and skips the rest. | Batch `[A(flag), B(flag), C(no flag)]`; B reverts. | State rolled back to before A; A's effects discarded; C skipped with `status=0x3`. | 📝 Planned |
| `test_atomic_batch_skipped_gas_refunded` | Gas allotted to skipped frames is refunded. | Batch where a middle frame fails, leaving later frames skipped. | Skipped frames' `gas_limit` refunded to payer; `status=0x3` for each. | 📝 Planned |
| `test_consecutive_atomic_batches` | Two batches are delimited by an unflagged frame. | `[A(flag), B(no flag), C(flag), D(no flag)]`; C reverts. | Batch1 `[A,B]` applied; batch2 `[C,D]` rolled back; D skipped. | 📝 Planned |

---

## 8. Introspection instructions

### 8.1 `TXPARAM` (`0xb0`)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_txparam_all_params` | Each defined `param` (`0x00`–`0x0b`) returns the correct transaction value. | Contract reads every param; tx with known nonce/sender/fees/blobs/frames/signatures. | Returned values match (type=`0x06`, nonce, sender, priority/max/blob fees, max cost, blob count, sig hash, frame count, current frame index, signature count); gas cost 2. | 📝 Planned |
| `test_txparam_current_frame_index` | `TXPARAM(0x0a)` reflects the executing frame's own index. | Same contract invoked from frames at different indices. | Returns each frame's own index. | 📝 Planned |
| `test_txparam_undefined_param_halts` | An undefined `param` causes an exceptional halt. | `TXPARAM(0x0c)`. | Exceptional halt. | 📝 Planned |

### 8.2 `FRAMEDATALOAD` (`0xb1`) / `FRAMEDATACOPY` (`0xb2`)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_framedataload_matches_calldataload` | `FRAMEDATALOAD(offset, frameIndex)` returns a word from the chosen frame's `data` (CALLDATALOAD semantics). | Frames with known data; read words at various offsets/indices. | Values match CALLDATALOAD over the target frame's data (zero-padded past end); gas cost 3. | 📝 Planned |
| `test_framedatacopy_matches_calldatacopy` | `FRAMEDATACOPY(memOffset, dataOffset, length, frameIndex)` copies into memory (CALLDATACOPY semantics + expansion). | Copy ranges within and past the end of a frame's data. | Memory matches CALLDATACOPY; gas = 3 + per-word + memory expansion. | 📝 Planned |
| `test_framedata_out_of_bounds_frameindex_halts` | Out-of-bounds `frameIndex` halts both ops (offset past `data` end does not). | `frameIndex = len(frames)`. | Exceptional halt. | 📝 Planned |
| `test_framedata_cross_frame_visibility` | Validation frames can read later `SENDER` frames' `data`. | `VERIFY` frame reads a subsequent `SENDER` frame's data. | Reads succeed and return the later frame's values. | 📝 Planned |

### 8.3 `FRAMEPARAM` (`0xb3`)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_frameparam_all_params` | Each `param` (`0x00`–`0x08`) returns the frame field. | Read resolved_target, gas_limit, mode, flags, len(data), allowed_scope, atomic_batch bit, value for a known frame. | Values match; `allowed_scope == flags & APPROVE_SCOPE_MASK`; `atomic_batch == (flags>>2)&1`; gas cost 2. | 📝 Planned |
| `test_frameparam_status_past_frame` | `FRAMEPARAM(0x05, i)` returns 0/1 for a completed earlier frame. | Later frame reads status of an earlier frame that succeeded/failed. | Returns 1 for success, 0 for failure. | 📝 Planned |
| `test_frameparam_status_current_or_future_halts` | Reading `status` of the current or a future frame halts. | `FRAMEPARAM(0x05, current)` and `FRAMEPARAM(0x05, current+1)`. | Exceptional halt in both. | 📝 Planned |
| `test_frameparam_out_of_bounds_and_undefined_halt` | Out-of-bounds `frameIndex` or undefined `param` halts. | `frameIndex = len(frames)`; `param = 0x09`. | Exceptional halt. | 📝 Planned |

### 8.4 `SIGPARAM` (`0xb4`)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_sigparam_all_params` | Each `param` (`0x00`–`0x03`) returns signature metadata; raw bytes are never exposed. | Tx with known signatures; read effective signer, scheme, msg, len(signature). | Values match; raw `signature` bytes not retrievable by any param; gas cost 2. | 📝 Planned |
| `test_sigparam_out_of_bounds_and_undefined_halt` | Out-of-bounds `signatureIndex` or undefined `param` halts. | `signatureIndex = len(signatures)`; `param = 0x04`. | Exceptional halt. | 📝 Planned |

---

## 9. Expiry verifier predeploy (`address(0x8141)`)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_expiry_verifier_code_installed` | The canonical runtime code is installed at `EXPIRY_VERIFIER` at activation. | Read code at `0x8141` post-fork. | Code == `0x60083614600a575f5ffd5b5f3560c01c4211601657005b5f5ffd`. | 📝 Planned |
| `test_expiry_verify_passes_before_deadline` | An `expiry_verify` frame succeeds when `block.timestamp <= expiry`. | `VERIFY` frame, target=`EXPIRY_VERIFIER`, flags=0, value=0, data = 8-byte big-endian expiry ≥ current timestamp. | Frame succeeds with no return data / no logs; tx proceeds. | 📝 Planned |
| `test_expiry_verify_reverts_after_deadline` | An `expiry_verify` frame reverts when `block.timestamp > expiry`. | 8-byte expiry < current timestamp. | `VERIFY` frame reverts → transaction invalid. | 📝 Planned |
| `test_expiry_verify_frame_constraints` | An expiry verifier frame is invalid unless `flags==0`, `value==0`, `len(data)==8` (a length mismatch is a frame-validity failure, not a runtime revert). | Each constraint violated individually. | Invalid for each violation. | 📝 Planned |
| `test_expiry_verify_at_most_one` | At most one expiry verifier frame is allowed. | Two frames targeting `EXPIRY_VERIFIER`. | Invalid. | 📝 Planned |
| `test_expiry_verifier_contract_direct_call` | The predeploy's runtime behaves correctly when called directly (outside a frame). | A contract `CALL`s `0x8141` with: (a) `calldatasize != 8`; (b) an 8-byte expiry ≥/< `TIMESTAMP`. | (a) reverts; (b) stops (success) when `timestamp <= expiry`, reverts when `timestamp > expiry`. | 📝 Planned |

---

## 10. Cross-frame semantics & `ORIGIN`

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_warm_cold_journal_shared_across_frames` | The warm/cold access journal is shared across frames. | Frame 1 touches account/slot X (cold→warm); frame 2 touches X and measures gas. | Frame 2 pays the warm cost for X. | 📝 Planned |
| `test_transient_storage_discarded_between_frames` | `TSTORE`/`TLOAD` transient storage is discarded between frames. | Frame 1 `TSTORE(k, v)`; frame 2 `TLOAD(k)`. | Frame 2 reads 0. | 📝 Planned |
| `test_origin_returns_frame_caller` | `ORIGIN` returns the frame's `caller` at all call depths. | `SENDER` frame (caller=sender) and `DEFAULT`/`VERIFY` frame (caller=ENTRY_POINT), each with nested calls reading `ORIGIN`. | `ORIGIN == sender` in SENDER frames; `ORIGIN == ENTRY_POINT` in DEFAULT/VERIFY frames, consistently across nesting. | 📝 Planned |

---

## 11. Sender is a contract account (EIP-3607 exemption)

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_sender_contract_account_allowed` | A frame tx whose `sender` has full contract code is valid (EIP-3607 not applied); `SENDER` frames originate from it. | `sender` has deployed smart-account code that approves and issues `SENDER` frames. | Transaction valid; `SENDER` frames execute with `caller == sender`. | 📝 Planned |

---

## 12. Blob (EIP-4844) integration

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_frame_tx_with_blobs` | A frame tx may carry `blob_versioned_hashes`; blob fees are added to max cost. | Non-empty blob hashes and non-zero `max_fee_per_blob_gas`. | `blob_fees = len(hashes) * GAS_PER_BLOB * blob_base_fee` included in `TXPARAM(0x06)`; `BLOBHASH` works in `SENDER` frames. | 📝 Planned |
| `test_frame_tx_blob_count_txparam` | `TXPARAM(0x07)` returns `len(blob_versioned_hashes)`. | Vary blob count. | Returned count matches. | 📝 Planned |

---

## 13. Integration flows

Worked examples not already covered mechanically above.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_example_1b_account_deployment` | EIP Example 1b: `[deploy, self_verify, user_op]` deploying a smart account at `sender`. | Frame 0 `DEFAULT` to the EIP-7997 factory with initcode+salt that installs code at `sender`; then `self_verify`; then `user_op`. | Code installed at `sender` before verification; verification then approves via the deployed code; the user op executes. | 📝 Planned |
| `test_example_3_sponsored_erc20` | EIP Example 3: sponsored tx paying fees in ERC-20 with a post-op. | `[only_verify(sender), pay(sponsor), user_op:transfer→sponsor, user_op:call, post_op(sponsor)]`. | `payer=sponsor`; sender approves execution; sponsor approves payment after inspecting the ERC-20 transfer frame; post-op runs; receipt lists sponsor as payer. | 📝 Planned |

---

## 14. Public mempool policy (deferred — non-consensus)

Propagation rules, not state transition — need a separate validation-prefix simulation harness.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_mempool_recognized_prefixes` | Only the four recognized validation-prefix shapes are accepted. | self_verify; deploy+self_verify; only_verify+pay; deploy+only_verify+pay (expiry frames skipped for matching). | Recognized prefixes accepted; others rejected. | ⏭️ Deferred |
| `test_mempool_max_verify_gas` | Validation-prefix gas + signature validation must not exceed `MAX_VERIFY_GAS` (100k). | Prefix whose gas exceeds the limit. | Rejected from public mempool. | ⏭️ Deferred |
| `test_mempool_banned_opcodes` | Banned opcodes in the validation prefix cause rejection (with the GAS-before-`*CALL` and deploy-frame exceptions). | Prefix using `TIMESTAMP`/`BALANCE`/`SSTORE`/etc. | Rejected; documented exceptions accepted. | ⏭️ Deferred |
| `test_mempool_state_access_rules` | Validation may only read `tx.sender` storage and existing-contract code; writes only inside a deploy frame to `tx.sender`. | Prefixes violating each trace rule. | Rejected per the validation trace rules. | ⏭️ Deferred |
| `test_mempool_canonical_paymaster` | A canonical paymaster is identified by exact runtime-code match; solvency via reservation. | `pay` frame targeting a canonical paymaster instance. | Admitted by code match + reservation; balance shortfall rejected. | ⏭️ Deferred |
| `test_mempool_non_canonical_paymaster_limit` | A non-canonical paymaster is limited to `MAX_PENDING_TXS_USING_NON_CANONICAL_PAYMASTER` (=1) pending txs. | Second pending tx sharing a non-canonical paymaster. | Second rejected. | ⏭️ Deferred |
| `test_mempool_expiry_drop` | A tx with an `expiry_verify` frame whose deadline has passed is dropped. | Expiry deadline < node's current block timestamp. | Dropped from public mempool. | ⏭️ Deferred |

---

## Notes for implementation

- **Fork:** targets **Bogota**, not yet defined in this repo (no
  `src/ethereum/forks/bogota`, no fork registration) — scaffold it before filling.
- **P256:** relies on the secp256r1 precompile (EIP-7951); gate those cases if absent.
- **EIP-7997:** deploy-frame cases assume its deterministic factory predeploy.
- **EIP-7819 `SETDELEGATE`:** referenced by deploy / banned-opcode rules; confirm presence.
- **EIP-7928 (BAL):** if co-shipped, add cases that BAL captures frame-tx accesses.
