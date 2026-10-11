# DGX full-S1 + harness-v4 closure, V31 static decision programs (pop-6848)

Input: registry snapshot `google-v31-source-closure-held-20261011-v1` (14 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 f986ba38…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 5d44497f. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v31-043211), fresh outputs:

- `raw-key-v31-s1.json`: Pop's exact raw key check (407 original row ids,
  407 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v31-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v31-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,628 keys) against the protected BGKit id map: no hits.
- `closure-v31-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 57/57 kept, all train, no external links, no protected hits, no check errors. Projection
  1674c1fd… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 f986ba38… (no `pins` map). `FILES.sha256` in the packet
directory verified completely (`sha256sum -c`, all OK), and every `{path, sha256}` pair in the packet matched: closure
inputs, Pop's raw-key harness check 80dd6fa2…, harness-v4 index files, and the hold policy
decision-visible-answerability-holds-v12 79b3ea6f…, which matches that file as committed in 9c10102d (exact copy in
`pinned-bytes/`). The packet ships no raw-key checker; the run used Pop's V29 `check_raw_group_keys.py` b945cbe8… (same
row schema; copy in `pinned-bytes/`). Shared closure code `source_case_closure.py` 64b255e8…. Ledger budget 5 GB
(sufficient; earlier closures peaked near 2–3 GB).
