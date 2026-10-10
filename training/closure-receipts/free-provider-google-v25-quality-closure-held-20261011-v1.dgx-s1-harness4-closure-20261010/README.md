# DGX full-S1 + harness-v4 closure, V25 static decision programs (pop-46ef)

Input: registry snapshot `free-provider-google-v25-quality-closure-held-20261011-v1` (21 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 a471a752…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 53fa56d6. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v25-010735), fresh outputs:

- `raw-key-v25-s1.json`: Pop's exact raw key check (402 original row ids,
  402 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v25-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v25-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,608 keys) against the protected BGKit id map: no hits.
- `closure-v25-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 53/53 kept, all train, no external links, no protected hits, no check errors. Projection
  b0622288… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 a471a752… (pins by Pop `/srv` paths, mapped to the synced
copies under `runs/free-provider-generation-20261009-v1/`). 23 of 25 pins were present and matched, including source
cases 98fdc126…, candidate actions ead45996…, source manifest aa842853…, Pop's `pop-harness4-v2` outputs, the raw-key
precheck, `check_raw_group_keys.py`, the parent review files, the held static adapter and the provider cases/labels.
Not checkable on DGX: `build_v25_closure_inputs.py` (pinned in the quality-review root but not part of the registered
snapshot) and the `training/decision_source_quality_holds.json` pin 4e7da13c…, which matches that file as committed in
a912a8ba (later commits changed the current policy). The packet does not pin the shared closure code; the run used
`source_case_closure.py` 64b255e8… (as V12–V21).
