# DGX full-S1 + harness-v4 exact-key closure, V36 static decision programs (pop-11f2)

Input: registry snapshot `google-choice-confidence-v36-source-closure-pop-held-20261011-v1` (9 files verified against its manifest on DGX), packet
`quality-review-v36-source-closure-packet-v2/closure-packet-v1.json` (sha256 6b563da3…, `FILES.sha256` d50e46b8…, every
listed file OK). Checker: the packet's `check_raw_group_keys.py` (sha256 b945cbe8…), run as the packet's
`suggested_check_command` from an origin/main worktree (ba99ed1a) under the memory ledger (5 GB, unit closure-v36-091931):

- `raw-key-v36-s1.json` (85e627ca65e8e03e…): 390 case rows (source cases sha256 1a264d8c…), 390 original row ids, 390 original
  group ids and namespace aliases against full S1 `cross-corpus-index-s1-full-final-20261003-v1/groups.json`
  (sha256 accb6981…, 985,434 keys): 0 row-id, 0 group-id, 0 alias hits.
- `raw-key-v36-harness4.json` (64ad5da5dc251883…): the same against harness-bench v4 `groups.json` (sha256 3d67d4c2…, 769 keys):
  0 / 0 / 0.
- `raw-key-v36-protected.json` (b4ac34e82d9922c0…, `raw-key-protected-check.py.txt`): original and namespaced ids/groups and
  their un-prefixed forms (1,560 keys) against the protected BGKit id map (sha256 585b21e7…): 0 hits.

V17 holds (`training/decision_source_quality_holds.json` c8eab8b4…, policy decision-visible-answerability-holds-v17):
the manifest's four held intersections (dc-74af9c378dd4f3634902, dc-3897b9f8e2f6503915d1, dc-4d14120a20f69c5871da,
dc-1d30afdaf80463b890d8 and their groups) lie in the 512-row prepared pool only; none of these ids or groups occurs in
`closure-source-cases.jsonl`, `selected-source-case-index.jsonl` or `selected-program-index.jsonl`. They stay held.

The packet carries no candidate records, so no shared `source_case_closure` sketch pass applies; this receipt is exact
keys only. Static programs are authored scaffolds, not model trajectories. Evidence only; held; no source-truth,
label-quality or training admission.
