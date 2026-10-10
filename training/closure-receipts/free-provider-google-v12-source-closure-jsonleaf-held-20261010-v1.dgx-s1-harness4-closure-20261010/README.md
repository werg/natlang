# DGX full-S1 + harness-v4 closure, V12 static decision programs (pop-5d39)

Input: registry snapshot `free-provider-google-v12-source-closure-jsonleaf-held-20261010-v1` (13 files verified
against its manifest on DGX), authoritative `closure-input-v4/` per Pop's `closure-packet-v1.json`; all packet pins
matched, including `check_raw_group_keys.py` and the shared `source_case_closure.py` (64b255e8…, origin/main fc844ccf).
Run from an origin/main worktree under the memory ledger (8 GB, unit closure-5d39-132519), fresh outputs:

- `raw-key-full-s1.json`: Pop's exact raw key check (389 original `dc-` row ids, 389 original `g-` group ids) against
  the full S1 `groups.json` (985,434 keys): 0 / 0 hits.
- `raw-key-harness4.json`: the same against harness-bench v4 groups: 0 / 0 hits.
- `raw-key-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their un-prefixed
  forms (1,556 keys) against the protected BGKit id map: 0 hits.
- `closure-full-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 51/51 kept, all train, no external links, no protected hits, no check
  errors. Projection a4202c61… is identical to Pop's harness4-v3 projection.

Report paths name DGX temp locations; hashes are authoritative. Same scope limit as Pop's packet: types.ts is not
projected. Evidence only; no source-truth, label-quality or training admission.
