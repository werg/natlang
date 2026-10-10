# DGX full-S1 + harness-v4 closure, V22 static decision programs (pop-9f15)

Input: registry snapshot `free-provider-google-v22-quality-held-20261010-v1` (16 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 17f23405…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 0bf1e327. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v22-221958), fresh outputs:

- `raw-key-v22-s1.json`: Pop's exact raw key check (394 original row ids,
  394 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v22-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v22-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,576 keys) against the protected BGKit id map: no hits.
- `closure-v22-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 53/53 kept, all train, no external links, no protected hits, no check errors. Projection
  cebcaf69… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Packet note: this packet uses the artifact schema (no `pins` map). Its 12 hashes (closure inputs, Pop's harness
outputs, harness report, raw-key precheck, `check_raw_group_keys.py`, `build_v22_closure_inputs.py`) and the parent
`../review.json` pin all matched. It does not pin the shared closure code; the run used `source_case_closure.py`
64b255e8…, the same version pinned by the V12–V21 packets.
