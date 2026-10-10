# DGX S1 + harness-v4 closure, Pop's ten static decision programs (pop-a058)

Input: registry snapshot `free-provider-google-choice-confidence-v9-source-admission-review-held-20261010-v1`
(verified against its manifest on DGX), `closure-input-v2/` files. Ran
`python -m scripts.neuralese_data.source_case_closure` from origin/main (worktree) under the memory ledger
(8 GB, unit closure-a058-110640) with `--index cross-corpus-index-s1-full-final-20261003-v1`,
`--index cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1` and the protected
BGKit map, fresh output directory. Files here are that output unchanged (`closure-report.json` paths name the
DGX temp locations; hashes are authoritative) plus `raw-alias-check.json`, the separate lookup of original raw
groups/ids and dataset record keys.

Result: `held-closed-no-cross-corpus-violations`; 10/10 records kept, all train; no external links, no
protected hits, no check errors; no raw alias hits (320 keys). Evidence only; no training admission.
