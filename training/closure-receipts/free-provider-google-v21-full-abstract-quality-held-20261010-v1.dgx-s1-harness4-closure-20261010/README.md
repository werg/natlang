# DGX full-S1 + harness-v4 closure, V21 static decision programs (pop-7ef3)

Input: registry snapshot `free-provider-google-v21-full-abstract-quality-held-20261010-v1` (23 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v2.json` (sha256 591b5ac6…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main fa546ac8. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v21-211121), fresh outputs:

- `raw-key-v21-s1.json`: Pop's exact raw key check (42 original row ids,
  42 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v21-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v21-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (168 keys) against the protected BGKit id map: no hits.
- `closure-v21-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 6/6 kept, all train, no external links, no protected hits, no check errors. Projection
  e48b210f… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Overlap accounting (requested in pop-7ef3): `overlap-accounting-v21.json`. V21 re-presents the six V18 v2 PubMedQA
programs with complete abstracts. All 42 original row ids and 42 original group ids are also in the V18 v2 closure
input (closed in `../free-provider-google-v18-quality-v2-held-20261010-v1.dgx-s1-harness4-closure-20261010/`); 42 of
V21's 44 projected source texts differ from V18's (full vs clipped abstracts). Across every .json/.jsonl under
`runs/free-provider-generation-20261009-v1` the keys also occur in the V18 v1 packet and static adapters, the V18/V22
provider queues, the V51 folder runtime and reviews and the PubMedQA full-abstract derivative; every split recorded
there is train; no held/test/validation conflict.
