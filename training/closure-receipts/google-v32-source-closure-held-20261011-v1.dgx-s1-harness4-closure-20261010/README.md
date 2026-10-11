# DGX full-S1 + harness-v4 closure, V32 static decision programs (pop-c1f1)

Input: registry snapshot `google-v32-source-closure-held-20261011-v1` (21 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 e3fa5c2f…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 47d715cf. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v32-051912), fresh outputs:

- `raw-key-v32-s1.json`: Pop's exact raw key check (393 original row ids,
  393 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v32-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v32-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,572 keys) against the protected BGKit id map: no hits.
- `closure-v32-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 53/53 kept, all train, no external links, no protected hits, no check errors. Projection
  8675aa53… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 e3fa5c2f… (no `pins` map). `FILES.sha256` verified completely,
and all 38 `{path, sha256}` pairs in the packet matched, including the shared checker code it pins
(`source_case_closure.py` 64b255e8…, `cross_corpus.py` 48355471…, `splits.py` decd51e3…, `records.py`,
`cross_corpus_registry.py`, `common/hashing.py`), the raw-key checker `check_raw_group_keys.py` b945cbe8… (the bytes
used), the current hold policy v13 (64 holds) and Pop's two harness runs. One pin names
`training/decision_source_quality_holds.json`; it matches that file as committed in ce136ba1. Ledger budget 5 GB.
