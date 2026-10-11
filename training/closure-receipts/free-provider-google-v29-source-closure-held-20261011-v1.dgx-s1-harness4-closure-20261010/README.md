# DGX full-S1 + harness-v4 closure, V29 static decision programs (pop-9f66)

Input: registry snapshot `free-provider-google-v29-source-closure-held-20261011-v1` (26 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 da9cc18e…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 3129564b. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v29-024144), fresh outputs:

- `raw-key-v29-s1.json`: Pop's exact raw key check (396 original row ids,
  396 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v29-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v29-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,584 keys) against the protected BGKit id map: no hits.
- `closure-v29-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 54/54 kept, all train, no external links, no protected hits, no check errors. Projection
  f9a5936c… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 da9cc18e…; its 14 path-carrying pins (closure inputs, Pop's
`pop-harness4-v1` raw-key and shared-closure outputs, execution-component manifest, builder 9aa965dd…, shared closure
code 64b255e8…, checker b945cbe8…) all matched on DGX.

Prior-nested reservations (pop-745c): `prior-reservation-closure-v29.json`. The V29 provider source manifest
(ecd9fa2c…) lists 36 prior provider batches and the V21 annotation reservations; all 37 files were present and
hash-matched. Every `dc-…`/`g-…` key referenced anywhere in them (a superset of direct program and nested source
ids/groups: 12,936 ids, 12,934 groups; the manifest's reserved unions are 13,224 ids / 12,934 groups, the id difference
being non-`dc-` keys) was intersected with V29's 396 original ids and 396 original groups: no hits.

Pinned bytes (`pinned-bytes/`, exact copies): the raw-key checker `check_raw_group_keys.py` b945cbe8… (stored as
`.py.txt`), the provider-preparation policy `decision_source_quality_holds.json` ea67230e… (= git bd360ed6), the static
current policy decision-visible-answerability-holds-v11 7dd8bcef… (= git 802a9e4d, 57 held items, 0 selected
intersections per the packet), and the DGX reservation script. Shared closure code `source_case_closure.py` 64b255e8…
as pinned.
