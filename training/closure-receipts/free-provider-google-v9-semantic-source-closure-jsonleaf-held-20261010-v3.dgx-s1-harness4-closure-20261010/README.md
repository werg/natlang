# DGX S1 + harness-v4 closure, ten static decision programs, JSON-leaf inputs (pop-6553)

Input: registry snapshot `free-provider-google-v9-semantic-source-closure-jsonleaf-held-20261010-v3` (verified
against its manifest on DGX), `closure-input-v3/` (source d9d4639e…, candidates f9e21e16…, closure manifest
963c2e97…). Ran `python -m scripts.neuralese_data.source_case_closure` at origin/main 2dd50a46 (includes the JSON-leaf
selection of 97e6b37e) under the memory ledger (8 GB, unit closure-6553-112058) against
`cross-corpus-index-s1-full-final-20261003-v1`, `cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1`
and protected BGKit, fresh output directory. Files here are that output unchanged (report paths name DGX temp
locations; hashes are authoritative).

Result: `held-closed-no-cross-corpus-violations`; 10/10 kept, all train; no external links, no protected hits, no
check errors. Projection identical to Pop's harness-only v3-final projection. The original raw-key check (320 keys,
no hits) in `../free-provider-google-choice-confidence-v9-source-admission-review-held-20261010-v1.dgx-s1-harness4-closure-20261010/raw-alias-check.json`
still applies: source and candidate files are unchanged. The v2 receipt is kept. Evidence only; no admission.
