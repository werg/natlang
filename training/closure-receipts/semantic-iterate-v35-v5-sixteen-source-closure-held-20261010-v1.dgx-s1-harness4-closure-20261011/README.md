# DGX full-S1 + harness-v4 closure, V35 V5 sixteen typed iterateOn train cases (pop-5586)

Input: registry snapshot `semantic-iterate-v35-v5-sixteen-source-closure-held-20261010-v1` (32 files verified against its
manifest on DGX); packet `packet.json` sha256 03610c6c…; all 31 `packet-files.json` hashes, the 6 roots (original 32
cases eb33efac…, selected 16 5de1be21…, results, 485 native turns, candidate manifest, source proof), the 5 pinned
closure scripts (source_case_closure 64b255e8…, cross_corpus, cross_corpus_registry, splits, records) at origin/main
d9383a3d, the registered S1 manifest 914893de… and its sketches.npz c2ae4b4a… (1.27 GB), protected BGKit 585b21e7… and
the packet's harness-v4 copy (identical to the registered index) all matched. Memory-ledger units (8 GB, then 4/8 GB
for the scans), fresh outputs.

- `closure-v35-s1-v1/`: Pop's command (DGX-RUN-INSTRUCTIONS.md) against the exact registered S1 sketches, harness v4 and
  protected BGKit: `held-closed-no-cross-corpus-violations`, 16/16 kept, all train, no external links,
  no protected hits, no check errors; internal near-duplicate links 16 (16 cases → 4 components, the four
  train domains). Projection 5c24258e… is identical to Pop's harness-only-closure-replay-v2 projection.
- `raw-key-v35.json`: the rows carry authored keys (no *_original fields); 36 direct keys (world, domain, source
  and case ids) plus 55 un-prefixed alias forms, against full S1 groups, harness v4 groups and protected ids:
  no hits.
- `overlap-world-accounting-v35.json`: the 32-source explicit split has 16 train / 16 test rows, 16 train and 16 test
  worlds, 4 train and 4 test domains, no group key shared across the split; the 16 selected rows are byte-identical to
  the original train rows. Registry-wide scan: 820 registered paths; json/jsonl files modified since
  2026-10-09 were content-scanned (11,547 files, 175 with V35 keys). Older files cannot contain V35 keys
  (authored 2026-10-10) and were not read; they are listed per path under `skipped_older_files_by_path`
  (508 paths, 1,924,377 files, 2933 GB). Its text-window split attribution is approximate; use the next file.
- `structured-split-occurrences-v35.json`: exact per-record split of every V35 world/domain key in the 175 hit files.
  The only disagreement with the current split is `runs/neuralese-semantic-self-improvement-v35-source-20261010-v1`:
  8 selected train worlds (EXPT-532/533, PROV-522/523, SAFE-512/513, SCHED-542/543) were test there and 8 current
  test worlds (COMP-550/551, IR-560/561, MAINT-580/581, TRIAGE-570/571) were train (`split-history-v35-v1-vs-v5.json`).
  That v1 is registered as `free-provider-v35-source-proposal-v1-excluded-held-20261010-v1` (excluded proposal,
  within-family cross split, superseded; no generation launched from it). Every other artifact (v2–v5 sources, Luna
  runtimes, reviews, audits, this packet) records each key with its current split. No held/test conflict.

Report paths name DGX temp locations; hashes are authoritative. Evidence only; held, no source-truth, semantic, action
or training admission (`training_approved` untouched).
