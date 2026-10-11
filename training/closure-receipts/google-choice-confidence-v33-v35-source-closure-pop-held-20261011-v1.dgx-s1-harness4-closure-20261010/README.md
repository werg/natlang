# DGX full-S1 + harness-v4 closure, V33/V35 static decision programs (pop-b5aa)

Input: registry snapshot `google-choice-confidence-v33-v35-source-closure-pop-held-20261011-v1` (15 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1/closure-packet.json` (sha256 9e6d4047…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main e40108e1. Run from an origin/main worktree under the memory ledger
(5 GB, unit closure-v33-v35-093150), fresh outputs:

- `raw-key-v33-v35-s1.json`: Pop's exact raw key check (824 original row ids,
  824 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v33-v35-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v33-v35-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (3,296 keys) against the protected BGKit id map: no hits.
- `closure-v33-v35-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 114/114 kept, all train, no external links, no protected hits, no check errors. Projection
  fd4252db… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

V16 holds: `decision-visible-answerability-root-review-v16/source-quality-holds.json` verified at sha256 69084484…
(root review de5b61f7…); none of its 150 held ids/groups occurs in `closure-source-cases.jsonl`, matching the
packet's zero exact ID/group intersection. The raw-key checker is Pop's `check_raw_group_keys.py` (b945cbe8…, the copy
shipped in the V37 packet, since this packet ships none). The 114 source cases carry 824 original row ids and 824
original group ids.
