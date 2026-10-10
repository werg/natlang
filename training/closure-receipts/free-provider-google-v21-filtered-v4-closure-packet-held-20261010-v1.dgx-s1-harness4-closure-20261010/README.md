# DGX full-S1 + harness-v4 closure, V21 filtered V4 static decision programs (pop-ac51)

Input: registry snapshot `free-provider-google-v21-filtered-v4-closure-packet-held-20261010-v1` (14 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 51dcca9e…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 00727455. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v21-filtered-v4-231924), fresh outputs:

- `raw-key-v21-filtered-v4-s1.json`: Pop's exact raw key check (37 original row ids,
  37 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v21-filtered-v4-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v21-filtered-v4-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (148 keys) against the protected BGKit id map: no hits.
- `closure-v21-filtered-v4-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 5/5 kept, all train, no external links, no protected hits, no check errors. Projection
  889c667a… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` (sha256 51dcca9e…) uses flat `closure_artifact_pins`; all 12 matched,
plus the corrected parent review pin `../review.json` c1e29664… (pop-83a5) and README 5cae92f9…. The packet does not pin
the shared closure code; the run used `source_case_closure.py` 64b255e8… (as pinned by V12–V21).

Overlap accounting against the earlier V21 receipt: `overlap-accounting-v21-filtered-v4.json`. V21 filtered V4 is a
quality-filtered subset of V21: all 37 original row ids and 37 group ids are in V21 (5 V21 rows filtered out, listed),
and all 39 projected source texts are byte-identical to V21's projection. The V21 cross-artifact scan (V18 v1/v2,
V18/V22 provider queues, V51, PubMedQA derivative; all train) covers every V4 key; no held/test/validation conflict.
