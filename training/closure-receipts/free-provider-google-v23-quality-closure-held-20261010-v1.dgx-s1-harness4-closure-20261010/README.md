# DGX full-S1 + harness-v4 closure, V23 static decision programs (pop-9ab0)

Input: registry snapshot `free-provider-google-v23-quality-closure-held-20261010-v1` (18 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 972ecbab…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 941e745c. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v23-233134), fresh outputs:

- `raw-key-v23-s1.json`: Pop's exact raw key check (397 original row ids,
  397 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v23-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v23-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,588 keys) against the protected BGKit id map: no hits.
- `closure-v23-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 53/53 kept, all train, no external links, no protected hits, no check errors. Projection
  ac73ab46… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` (sha256 972ecbab…) uses flat `artifact_pins`; all 12 matched (source
cases c4413a36…, candidate actions 3a1ef236…, source manifest 33f64f68…, Pop harness outputs/report, raw-key precheck,
`check_raw_group_keys.py`, `build_v23_closure_inputs.py`), plus README f80cae42… and the parent `../review.json`
d8b8d104…. The raw-key check covers the 397 direct original row ids and 397 group ids as well as their namespace
aliases. The packet does not pin the shared closure code; the run used `source_case_closure.py` 64b255e8… (as pinned
by V12–V21).
