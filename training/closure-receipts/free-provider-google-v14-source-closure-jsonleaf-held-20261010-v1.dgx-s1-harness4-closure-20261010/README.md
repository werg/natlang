# DGX full-S1 + harness-v4 closure, V14 static decision programs (pop-33b6)

Input: registry snapshot `free-provider-google-v14-source-closure-jsonleaf-held-20261010-v1` (14 files verified
against its manifest on DGX), authoritative `closure-input-v1/` per Pop's `closure-packet-v1.json` (sha256 328afa5f…);
all packet pins matched, including `check_raw_group_keys.py`; shared `source_case_closure.py` 64b255e8… at origin/main
0eacaf69. Run from an origin/main worktree under the memory ledger (8 GB, unit closure-33b6-142519), fresh outputs:

- `raw-key-v14-s1.json`: Pop's exact raw key check (368 original row ids, 368 original group ids, namespace aliases)
  against the full S1 `groups.json` (985,434 keys): no hits.
- `raw-key-v14-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v14-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,472 keys) against the protected BGKit id map: no hits.
- `closure-v14-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 47/47 kept, all train, no external links, no protected hits, no check
  errors. Projection 7cc0ce23… is identical to Pop's closure-harness4-v1 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.
