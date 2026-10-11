# DGX full-S1 + harness-v4 closure, V73 native candidate subset (pop-8b3f)

Input: registry snapshot `v73-native-candidate-s1-closure-request-held-20261011-v1` (13 files verified against its manifest on DGX), packet
`quality-review-v73-native-candidate-s1-closure-v1/closure-packet-v1.json` (sha256 6627cd11…, `FILES.sha256` 000c278a…).
Authoritative inputs are `closure-input-v2/` as pinned in the packet's `source_inputs`; every pin matched: source cases
2af0babf… (43 programs), candidate turns 76dbc99f… (909 rows), source manifest d51ee3ae…, raw results c384b561…,
candidate manifest 13b4918b…, quality review e2201e50…, source labels 9118d743…, operational metrics 644086d5…,
policy `training/decision_source_quality_holds.json` c8eab8b4… (= main). `FILES.sha256` also lists the superseded
`closure-input-v1/`, `build_inputs.py`, `build-summary.json`, `operational-metrics-v1.json` and `FILES-v2.sha256`, which
are neither registered nor synced to DGX; the v1 inputs share the v2 hashes except the source manifest. Run from an
origin/main worktree (08efd703) under the memory ledger (5 GB, unit closure-v73-112443), fresh outputs:

- `raw-key-v73-s1.json` (22d881b253543db0…): Pop's `check_raw_group_keys.py` (b945cbe8…; the packet ships none)
  over 43 case rows: 302 original row ids, 302 original group ids and namespace aliases against full S1 `groups.json`
  (accb6981…, 985,434 keys): 0 / 0 / 0.
- `raw-key-v73-harness4.json` (c8372b3dc366e77a…): the same against harness-bench v4 `groups.json`
  (3d67d4c2…, 769 keys): 0 / 0 / 0.
- `raw-key-v73-protected.json` (a02fa1500c0f84d2…, `raw-key-protected-check.py.txt`): original and
  namespaced ids/groups and un-prefixed forms (1,208 keys) against the protected BGKit id map (585b21e7…): 0 hits.
- `closure-v73-s1-v1/` (report 6f7fa0688b890390…): `python -m scripts.neuralese_data.source_case_closure`
  (64b255e8…, cross_corpus.py 48355471…) `--source closure-source-cases.jsonl --candidate-rows closure-candidate-turns.jsonl
  --source-manifest closure-source-manifest.json --index cross-corpus-index-s1-full-final-20261003-v1` (manifest 914893de…,
  index eb78bbdf…, 1,870,591 records, full sketches) `--index` harness v4 (manifest 18bf74c8…) `--protected`:
  `held-closed-no-cross-corpus-violations`, 43/43 kept, all train, no external links, no internal split/group errors, no
  cross-corpus check errors, 0 protected hits in all 43 cases. Projection 49698719… is identical to Pop's `pop-harness4-v2`.

V17 holds: none of the 158 held ids/groups occurs in the source cases or the 909 candidate turns.

Report paths name DGX temp locations; hashes are authoritative. This binds the reviewed v2 909-turn packet only (not the
V73 v3 action-failure projection). Evidence only; held; no native, trace, source-quality or training admission; no
new-world credit.
