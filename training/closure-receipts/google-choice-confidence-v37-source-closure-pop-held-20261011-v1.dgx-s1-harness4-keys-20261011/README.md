# DGX full-S1 + harness-v4 exact-key closure, V37 static decision programs (pop-e094)

Input: registry snapshot `google-choice-confidence-v37-source-closure-pop-held-20261011-v1` (9 files verified against its manifest on DGX), packet
`quality-review-v37-source-closure-packet-v1/closure-packet-v1.json` (sha256 d494a03a…, `FILES.sha256` 84cd1ea9…, every
listed file OK). Checker: the packet's `check_raw_group_keys.py` (sha256 b945cbe8…), run as the packet's
`suggested_check_command` from an origin/main worktree (05cb694d) under the memory ledger (5 GB, unit closure-v37-092201):

- `raw-key-v37-s1.json` (0ba634b6645fa3d7…): 414 case rows (source cases sha256 3c7df72f…), 414 original row ids, 414 original
  group ids and namespace aliases against full S1 `cross-corpus-index-s1-full-final-20261003-v1/groups.json`
  (sha256 accb6981…, 985,434 keys): 0 row-id, 0 group-id, 0 alias hits.
- `raw-key-v37-harness4.json` (20e20aab2747b897…): the same against harness-bench v4 `groups.json` (sha256 3d67d4c2…, 769 keys):
  0 / 0 / 0.
- `raw-key-v37-protected.json` (6792062185fdf68e…, `raw-key-protected-check.py.txt`): original and namespaced ids/groups and
  their un-prefixed forms (1,656 keys) against the protected BGKit id map (sha256 585b21e7…): 0 hits.

V17 holds: the packet's `current-source-quality-holds.json` is byte-identical to `training/decision_source_quality_holds.json`
(c8eab8b4…, decision-visible-answerability-holds-v17) on main; none of its 158 held ids/groups occurs in
`closure-source-cases.jsonl`, matching the manifest's empty `held_intersections`.

The packet carries no candidate records, so no shared `source_case_closure` sketch pass applies; this receipt is exact
keys only. Static programs are authored scaffolds, not model trajectories; V38 is not covered. Evidence only; held; no
source-truth, label-quality or training admission.
