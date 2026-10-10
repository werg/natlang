# DGX full-S1 + harness-v4 closure, V20 static decision programs (pop-0b83)

Input: registry snapshot `free-provider-google-v20-quality-held-20261010-v1` (17 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 2e9312f0…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main fa546ac8. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v20-210956), fresh outputs:

- `raw-key-v20-s1.json`: Pop's exact raw key check (408 original row ids,
  408 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v20-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v20-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,632 keys) against the protected BGKit id map: no hits.
- `closure-v20-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 54/54 kept, all train, no external links, no protected hits, no check errors. Projection
  faf44e80… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.
