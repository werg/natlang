# DGX full-S1 + harness-v4 closure, V27 static decision programs (pop-4634)

Input: registry snapshot `free-provider-google-v27-quality-closure-held-20261011-v1` (18 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 47d1bcc2…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 802a9e4d. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v27-022154), fresh outputs:

- `raw-key-v27-s1.json`: Pop's exact raw key check (404 original row ids,
  404 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v27-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v27-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,616 keys) against the protected BGKit id map: no hits.
- `closure-v27-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 54/54 kept, all train, no external links, no protected hits, no check errors. Projection
  b1db9d3f… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 47d1bcc2… (pins by Pop `/srv` paths, mapped to the synced
copies under `runs/free-provider-generation-20261009-v1/`). 26 of 27 pins were present and matched, including source
rows 0e9e7760…, candidate actions 33293fbe…, source manifest 6f880df4…, Pop's `pop-harness4-v1` outputs, the raw-key
precheck and `check_raw_group_keys.py`. The `training/decision_source_quality_holds.json` pin ea67230e… (the policy
loaded by the static adapter, 45 items at batch seal) matches that file as committed in bd360ed6. The packet does not
pin the shared closure code; the run used `source_case_closure.py` 64b255e8… (as V12–V25).
