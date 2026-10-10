# DGX full-S1 + harness-v4 closure, V15 static decision programs (pop-4343)

Input: registry snapshot `free-provider-google-v15-source-closure-jsonleaf-held-20261010-v1` (14 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 3594852f…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 91156e6c. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v15-153044), fresh outputs:

- `raw-key-v15-s1.json`: Pop's exact raw key check (376 original row ids,
  376 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v15-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v15-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,504 keys) against the protected BGKit id map: no hits.
- `closure-v15-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 49/49 kept, all train, no external links, no protected hits, no check errors. Projection
  91e72013… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.
