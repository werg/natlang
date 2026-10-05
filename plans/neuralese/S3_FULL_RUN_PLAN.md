# S3 full run: plan

Draft, 2026-10-05. Follows [S3_FULL_RUN_HANDOFF.md](S3_FULL_RUN_HANDOFF.md) (prerequisites) and [S3_PORT.md](S3_PORT.md)
(phases A–F). The run needs the DGX's GPU to itself for about a day: **the owner decides when to pause the teacher
campaign**; nothing here stops it.

## 1. Evidence: pilot v4

`runs/neuralese-s3-pilot-20261005-v4` (LFM2.5-350M, cutoff 6, max length 32, Muon, families qa_extractive /
qa_multihop / tool_digest from the closed samples, ~2,000 training records each). New against the EF pilot: variable
span lengths in A–C (`--span-lengths 8,12,16,24,32`), phase-D lengths from source size (`--tokens-per-vector 16`), the
full-depth stop source (`--stop-source final`), stop exploration in E (`--stop-exploration 0.5`), the shortcut gate
(`--fail-on-shortcut`). A run with 4 tokens per vector was stopped: nearly every source reached the 32-vector cap.

Held-out records after D (EF pilot → v4; correct − shuffled consumer NLL in nats, lower is better):

| | EF pilot | v4 |
| --- | --- | --- |
| qa_extractive | −0.116 | −0.187 |
| qa_multihop | −0.091 | −0.136 |
| tool_digest | −0.010 | −0.038 |
| block lengths | all 16 | 25 distinct, mean 21.6 |
| Spearman(length, source tokens) | — | 0.92 |
| truncated at the cap | 0 | 0.42 |
| effective rank of payloads | 80 | 265 |
| cross-source similarity | 0.40 | 0.34 |

The written blocks now carry more source-specific content and their length follows the source. 42% of records hit
the 32-vector cap: sources longer than 512 tokens cannot be represented at 16 tokens per vector within it.

After E and F: see §7.

## 2. Data

S1 final (`port-records/full-20261003/resume-20261003T1948-review1/final/`, audited: schema, structure, leakage,
duplicate IDs, protected splits): 1,870,591 records, 45 families, 1,361,548 train / 52,439 validation / 456,604
test. Five families have no training split (qa_extractive, qa_babi_filler, qa_needles, qa_needles_long,
qa_extractive_article): they are evaluation families. The largest training families are trajectory continuations
(terminal 323k, research 142k, SWE 135k).

Built (2026-10-05): `scripts/neuralese_s3_subset.py` → `/home/werg/data/neuralese-s3-full/records/` (NVMe, 21 GB,
`manifest.json`): 45 families, 569,206 training records (11 families at the cap) and 16,429 evaluation rows (the
first 400 per family; the harness takes 64 that fit). Two training families have no evaluation split in S1
(`trajectory_continuation_search`, `trajectory_continuation_terminal`: every record is `train`), so the harness does
not cover them.

Sampling: per family, at most 25,000 training records (deterministic by source group; `--stream` loader with resume
position), so no family dominates: about 520k records. Held-out: each family's validation or test split, 64 records
per family for the harness, never trained.

## 3. Schedule

One cutoff (6). Phases as the pilot, scaled: A 3k, B 3k, C 4k, D 30k, E 4k, F 6k steps (batch 8). D carries the
content: at the pilot's 5.1 s per step with the GPU shared, D alone would take 43 h; with the campaign paused
expect roughly 2–2.5× faster (to be measured in the first hour), so the run takes about one day.

Settings changed from the pilot:

- `--max-length 64` (from 32) and `--tokens-per-vector 16`: sources up to 1,024 tokens are represented without
  truncation; longer ones are truncated at 64 and reported. Inference has no length budget (the stop head decides,
  or a caller's hint): the maximum is a runaway bound.
- Everything else as v4: variable span lengths, final stop source, stop exploration 0.5, Muon with the AdamW
  split for embeddings/readouts, shortcut gate.

Checkpoints at every phase boundary and every 2,000 steps, with optimizer state, sampler position and RNG state.

## 4. Evaluation (continuous, no fixed thresholds)

At each phase boundary: the pilot harness on the held-out records of every family (correct / shuffled / zeroed /
no block / full text, gap recovered, length histogram, Spearman with source size, truncation, effective rank,
cross-source similarity, cache agreement, latency), plus:

- export to GGUF and the server conformance suite (both servers, native and wasm) on the trained heads;
- the trajectory trainer's smoke (`train.trajectories --handover written`) with the new writer: the written notes
  and child results must beat their shuffled versions in their readers;
- D stage 2 (`delta_e2e`, written codes): zero-shot code against another family's code, the check the pilot writer
  failed (its codes carried no family-specific information);
- ordinary Natlang execution replay on the student (`execution-eval-v3`), so port training does not cost text skill.

## 5. Memory

The pilot ran inside a 24 GB CUDA cap (it ran out of memory at 14 GB in D). At max length 64 and batch 8, budget
48 GB through the ledger (`--memory-gb 44`). With the campaign paused the DGX has ~110 GB free.

## 6. Same window: jobs that need the campaign paused

- Clef-flash decision labels (`scripts/label_decision_cases.py --backend clef`): 18 GB of bf16 weights, but loading
  peaks at 41.8 GB (a host copy plus the GPU copy in unified memory; steady state ~24 GB). With the campaign's
  ~59 GB resident the ledger cannot admit it with its reserve (the guard stopped a 36 GB attempt at 41.8 GB).
  Script: `.../tmp/clef.sh` pattern, budget 48 GB.

## 7. Pilot v4 after E and F

Finished (wall 15.6 min for E+F after two fixes, peak GPU 15.2 GB). Held-out records after F (EF pilot after F →
v4 after F; correct − shuffled, nats): qa_extractive −0.132 → −0.067, qa_multihop −0.124 → −0.159, tool_digest
−0.022 → −0.036; spans −0.027 → −0.102. Lengths: 22 distinct, Spearman 0.94 with source size, 37% at the cap. In E
the exploration finally varies the stop (batch length std 8.9; the EF pilot stopped at 16 in 197 of 200 batches),
and training batches show correct beating shuffled by 1.5–2.0 nats (EF: ~0.8–1.2).

Watch in the full run: qa_extractive lost content use from D (−0.187) to F (−0.067) while the others held or gained;
F's LoRA and text replay may trade it away. Keep the per-family harness at every phase boundary and compare D and F
per family before accepting F.

Two trainer bugs surfaced by variable lengths and fixed (with tests): the shuffled contrast's negatives did not fill
the padded payload width under sampled stops (E), and text replay stacked spans of different lengths (F).

## 8. Launch

`scripts/neuralese_s3_full_run.sh [RUN]` runs the schedule above (`--phase-steps A=3000,B=3000,C=4000,D=30000,E=4000,F=6000`,
ramps scaled with their phases; `--harness-phases C,D,E,F` (A and B come before the port carries content; each
harness takes about an hour at 64 records per family); `--checkpoint-every 500`; `checkpoint-after-X.pt`
kept at each phase boundary; evaluation-only families in the harness through `--eval-families`). Rerunning with the
same RUN resumes. Through the ledger, once the campaign is paused:

    python3 scripts/memory_ledger.py run --unit natlang-s3-full --budget-gb 48 --class experiment --wait 3600 \
      --workdir /home/werg/natlang -- scripts/neuralese_s3_full_run.sh

Rehearsal (2026-10-05, beside the campaign; `/home/werg/data/neuralese-s3-rehearsal2`, removed): the launch script
with A=150, B=150, C=200, D=40 (batch 4), E=4, F=4 steps ran end to end on the subset: stream index over 45
families (6 min), 43 evaluation families in the harness, writes up to 64 vectors (D block lengths 28–51), boundary
checkpoints kept, resume after a stop. Gradient norms (median, pre-clip): A 5.6, B 2.9, C 6.3, D 17.7 (pilot v4 D
11.8). An earlier rehearsal with only 14 steps of A–C saw D norms of 1e6–1e13: the heads must be trained before
long writes. Memory: D at length 64 and batch 8 does not fit in 30 GB of CUDA (it ran out beside the campaign's
60 GB); budget 48 GB as planned, with the GPU to itself. `--phase-batch D=4` exists for constrained rehearsals.
