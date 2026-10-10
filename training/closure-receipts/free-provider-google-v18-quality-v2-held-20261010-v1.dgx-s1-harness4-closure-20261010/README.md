# DGX full-S1 + harness-v4 closure, V18 static decision programs (pop-a01c)

Input: registry snapshot `free-provider-google-v18-quality-v2-held-20261010-v1` (0 files verified against its manifest on DGX), authoritative `closure-input-v2/` per Pop's
`closure-packet-v2.json` (sha256 326016ed…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 9a44436e. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v18-191912), fresh outputs:

- `raw-key-v18-s1.json`: Pop's exact raw key check (399 original row ids,
  399 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v18-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v18-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,596 keys) against the protected BGKit id map: no hits.
- `closure-v18-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 53/53 kept, all train, no external links, no protected hits, no check errors. Projection
  2b80be0e… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.
