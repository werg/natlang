# DGX full-S1 + harness-v4 closure, V64 seven-source related recovery (pop-5ef1)

Input: registry snapshot `luna-v64-seven-source-quality-closure-held-20261011-v1` (verified on DGX); packet `packet.json`
sha256 61c7e3fa…. The packet binds 7 DBpedia source rows (fancyzhx/dbpedia_14, cc-by-sa-3.0, role train) by row index and
raw-row SHA-256 in the master decision-v1 snapshot (sha256 4fea381b…, local on DGX); the rows were rebuilt from that
snapshot and every raw-row hash and id matched (`row_checks`). The recovery static directory itself is not synced; the
exact master rows make it unnecessary for closure.

Method (`v64-closure.py.txt`, run from an origin/main worktree ce136ba1 under the memory ledger, unit
closure-v64-043020, fresh output): per row, split-group lookup of id, group and source:id; exact normalized-text digests
of state, question and state+question; question/example digests (best-effort); MinHash near-duplicates of the state
(Jaccard ≥ 0.8) against `cross-corpus-index-s1-full-final-20261003-v1` (full sketches) and
`cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1`; protected BGKit (585b21e7…)
question hashes and ids. The two program IRs: program id, source ids and source groups against both indexes. The same
lookups were positive-controlled on published S1 records in the static16 receipt. Code hashes are recorded in the
result; `cross_corpus.py` 48355471… and `splits.py` decd51e3… equal the checker hashes Pop pinned.

Result (`closure-v64-seven-source.json`): all 7 rows and both programs closed: no group, exact-text, question,
example or near-duplicate hit in S1 or harness v4, no background-source hit, no protected question or id hit. Original
source license and split (train) are recorded per row. Evidence only; held; no source-quality eligibility, trace or
training admission; no IR or flag changes.
