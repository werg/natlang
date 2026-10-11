# DGX full-S1 + harness-v4 closure, V30 static decision programs (pop-9f38)

Input: registry snapshot `free-provider-google-v30-source-closure-held-20261011-v1` (14 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 0f7b25d7…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 191792b7. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v30-034815), fresh outputs:

- `raw-key-v30-s1.json`: Pop's exact raw key check (403 original row ids,
  403 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v30-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v30-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,612 keys) against the protected BGKit id map: no hits.
- `closure-v30-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 56/56 kept, all train, no external links, no protected hits, no check errors. Projection
  b80c595d… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 0f7b25d7… carries no `pins` map; every `{path, sha256}` pair in
it (closure inputs: source cases 2c92af0a…, candidate rows 95354685…, both indexes; Pop's `pop-harness4-v1` projection
b80c595d…, report; execution-component manifest 6674905e…) matched, and the source manifest is 74656b5f… as quoted.
The packet ships no raw-key checker; the run used Pop's V29 `check_raw_group_keys.py` b945cbe8… (same row schema;
copy in `pinned-bytes/`, which also holds the policies the packet names: provider policy 7dd8bcef… = git 802a9e4d,
canonical current policy 79b3ea6f… = git 9c10102d, V12 additive five-hold receipt 4692cb5d…). The packet does not pin
the shared closure code; the run used `source_case_closure.py` 64b255e8… (as V12–V29).
