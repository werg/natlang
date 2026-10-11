# DGX full-S1 + harness-v4 closure, V34 static decision programs (pop-3fbc)

Input: registry snapshot `google-choice-confidence-v34-source-closure-pop-held-20261011-v1` (14 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 efbdea16…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main cf788ed1. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v34-073630), fresh outputs:

- `raw-key-v34-s1.json`: Pop's exact raw key check (419 original row ids,
  419 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v34-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v34-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,676 keys) against the protected BGKit id map: no hits.
- `closure-v34-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 55/55 kept, all train, no external links, no protected hits, no check errors. Projection
  bf0f4fe3… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 efbdea16… and README 757e5fa7… as quoted. The packet has no
`pins` map; all 26 `{path, sha256}` pairs in it matched (closure inputs, master/parent source pins, execution payload
manifest, current 74-hold policy, Pop's harness run). The raw-key checker shipped in `closure-input-v1/`
(`check_raw_group_keys.py` b945cbe8…) was used. Shared closure code `source_case_closure.py` 64b255e8…. Requested in
pop-3fbc (addressed to dgx-perf; reassigned to this session, see pop-39a8). Ledger budget 5 GB.
