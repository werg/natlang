# DGX full-S1 + harness-v4 exact-key closure, V74 static annotation-folder source set (pop-5c7d)

Input: registry snapshot `luna-v74-exact-source-closure-request-held-20261011-v1` (9 files verified against its manifest on DGX), packet
`quality-review-v74-source-closure-packet-v5/closure-packet-v1.json` (sha256 80f269ed…, `FILES.sha256` a2251fbd…, every
listed file OK). Checker: the packet's `check_raw_group_keys.py` (sha256 b945cbe8…), run as the packet's
`suggested_check_command` from an origin/main worktree (18e12945) under the memory ledger (5 GB, unit closure-v74-101145):

- `raw-key-v74-s1.json` (8035fcc92579070c…): 640 case rows (source cases sha256 8bf30f5d…), 640 original row ids,
  640 original group ids and namespace aliases against full S1 `cross-corpus-index-s1-full-final-20261003-v1/groups.json`
  (sha256 accb6981…, 985,434 keys): 0 row-id, 0 group-id, 0 alias hits.
- `raw-key-v74-harness4.json` (b601f160fc8b5cc2…): the same against harness-bench v4 `groups.json`
  (sha256 3d67d4c2…, 769 keys): 0 / 0 / 0.
- `raw-key-v74-protected.json` (8803a675b6362146…, `raw-key-protected-check.py.txt`): original and
  namespaced ids/groups and their un-prefixed forms (2,560 keys) against the protected BGKit id map (sha256 585b21e7…): 0 hits.

V17 holds: the packet's `current-source-quality-holds.json` is byte-identical to `training/decision_source_quality_holds.json`
on main (c8eab8b4…); none of its 158 held ids/groups occurs in `closure-source-cases.jsonl`, matching the packet's zero
intersection count.

Exact keys only: the packet carries no candidate records, so no shared `source_case_closure` sketch pass applies. The
80 static programs are authored scaffolds, not model trajectories. Evidence only; held; no source-quality clearance or
training admission.
