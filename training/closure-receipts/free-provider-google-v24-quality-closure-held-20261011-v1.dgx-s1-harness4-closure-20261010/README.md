# DGX full-S1 + harness-v4 closure, V24 static decision programs (pop-1106)

Input: registry snapshot `free-provider-google-v24-quality-closure-held-20261011-v1` (23 files verified against its manifest on DGX), authoritative `closure-input-v1/` per Pop's
`closure-packet-v1.json` (sha256 8446d6d9…); all packet pins matched, including `check_raw_group_keys.py`; shared
`source_case_closure.py` 64b255e8… at origin/main 409e8e2d. Run from an origin/main worktree under the memory ledger
(8 GB, unit closure-v24-012908), fresh outputs:

- `raw-key-v24-s1.json`: Pop's exact raw key check (406 original row ids,
  406 original group ids, namespace aliases) against the full S1 `groups.json`
  (985,434 keys): no hits.
- `raw-key-v24-harness4.json`: the same against harness-bench v4 groups: no hits.
- `raw-key-v24-protected.json` (`raw-key-protected-check.py`): original and namespaced ids/groups and their
  un-prefixed forms (1,624 keys) against the protected BGKit id map: no hits.
- `closure-v24-s1-v1/`: shared closure against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches),
  `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and protected BGKit:
  `held-closed-no-cross-corpus-violations`, 54/54 kept, all train, no external links, no protected hits, no check errors. Projection
  92b486f6… is identical to Pop's closure-harness4 projection.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; no source-truth, label-quality or
training admission.

Inputs and pins: packet `closure-packet-v1.json` sha256 8446d6d9… (pins by Pop `/srv` paths, mapped to the synced
copies under `runs/free-provider-generation-20261009-v1/`). 24 of 25 pins were present and matched, including source
cases f3ed09dd…, candidate actions d49522b7…, source manifest 94d634a1…, Pop's `pop-harness4-v2` outputs, the raw-key
precheck, `check_raw_group_keys.py` and the builder. The `training/decision_source_quality_holds.json` pin 7fbae8f0…
(the live policy Pop cited, 30 ids) matches that file as committed in 796b1339. The packet does not pin the shared
closure code; the run used `source_case_closure.py` 64b255e8… (as V12–V25).
