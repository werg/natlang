# Neuralese implementation backlog — 2026-10-09

Audit of the planned deliverables in README.md (S0–S8), S5–S8, LEARNING_CONTINUUM.md, MAPLE_NESTED.md,
BATCHED_EXECUTION.md §7 and HANDOVER.md against the code on `main` (628fc90c). Classes:

- **used**: built and exercised by a current run or test path
- **not wired**: built, but no current pipeline (Maple/Mellum line) uses it
- **partial**: some of the deliverable exists
- **not started**: nothing beyond the plan

The main line moved from LFM2.5-350M to Maple and then Mellum (2026-10-09). Most S4–S6 machinery was built and
exercised on the 350M port and the Python reference server. "Not wired" mostly means "not yet on Mellum".

## S0 spec and S1 data

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Neuralese chapter, types, `.nz`, dialect tags, graph record | used | `spec/SPEC.md`, `spec/NEURALESE_*.md`, `ts-host/src/native/nz-file.ts`, `natlang_neuralese/nz.py` |
| Port records, ledger, splits | used | `scripts/neuralese_port_records.py`, `training/neuralese_data_ledger.json` |
| Corpus registry with SHA-256 manifests | used | `training/neuralese_corpora.json`, `training/corpus-manifests/`, `scripts/sync_training_corpora.py` |
| v13 soft conversion (system prompts, handovers, pinned notes as soft parts) | used | `ts-host/scripts/neuralese-convert-trajectories.mjs`; the converted `records`/`pieces` format is what `train/trajectories.py` trains on (soft parts become leaves, `--bank` initialises prompt pieces). Correction: the first version of this audit said "consumed by no trainer"; that note in HANDOVER predates e10363b5 |
| BGKit/Schnitzeljagd → NatLang program families (#41) | not started | — |

## S2 skill authoring

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Skill assets, pool, scope injection, disclosure | used | `ts-host/src/skills/` |
| Episodes, gate, improvement-step records | used | `ts-host/src/improvement/step-record.ts`, `scripts/episode_lib.py`, `ts-host/scripts/skills/` |
| Crisp student trained on the authoring corpus | partial | Corpora exist; no Mellum training stage consumes them |

## S3 port

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Foundation warm-up, runtime check, raw recurrence (Maple) | used | `train/text_warmup.py`, `train/trajectories.py`, recipes |
| Mellum port, markers, parity, full-latent QAT conversion | used (trial running) | `maple/model.py`, `maple/qat_convert.py`, `plans/mellum-port.md` |
| Mellum nested family N0–N2 (#44) | not started | `maple/family.py` exists for Maple only |
| Mellum foundation warm-up → runtime → recurrence (#45) | not started | No Mellum recipe in `training/neuralese/recipes/` |
| Compression stage (tokens_per_vector > 1) | partial | Options exist in `trajectories.py`; no recipe or gate |
| Neuralese autoregressive block layout switch (#42) | not started | — |
| Dialect tag for the Mellum space | not started | Defect: raw-token heads report `nd:natlang-raw-token@1` for every backbone (`model/heads.py:409`), so LFM (width 1024), Maple and Mellum (2304) spaces share one tag although a tag names the width (spec/NEURALESE_DIALECTS.md). The artifact registry pins the backbone as a guard; the tag itself needs an owner decision (it changes frozen runs' identity) |

## S4 runtime and servers

| Deliverable | Class | Evidence |
| --- | --- | --- |
| TS runtime: type, literals, `read`, contexts, `.nz`, graph | used | `ts-host/src/native/`, `ts-host/src/neuralese/` |
| Python reference server incl. `grad`, `decide`, batched score | used | `serve/` (decide_many 628fc90c) |
| Reference server on Maple ports | used | `serve/__init__.py` (`backbone == "maple"`) |
| Reference server on Mellum (markers, QAT latents) | partial | Markers via `hf_port.family_controls`; QAT conversion output not loadable as a served student yet |
| llama.cpp fork (LFM2): ports, server, LoRA adapters, read adapter | used | fork 8155cff42, `tests/neuralese/test_server_conformance.py` |
| llama.cpp fork for Maple/Mellum (MoE, ternary TQ2_0, heads) (#26, #32) | not started | `export/` handles LFM2 only |
| Browser wasm/WebGPU | used (LFM, lagging fork build) | `ts-host/vendor/neuralese-wasm/provenance.json` (fork 5d999c0c5) |
| vLLM rollouts (text on vLLM, writes on reference) | partial | `scripts/neuralese_vllm_parity.py`; no Mellum plugin |
| Batched-execution integration (BATCHED_EXECUTION §7: diagnostic code union, turn/function in requests, shared scheduler in improvement/pi drivers, author note) | not started on main | Owner: TS batching session |

## S5 program-level training

| Work item (S5 §12) | Class | Evidence |
| --- | --- | --- |
| 1 Consumer tracing over graph records | not started | No consumer classifier; `native/graph.ts` records nodes |
| 2 Compiler conversion pass (value/function/skill sites) | partial | `neuralese-rewrite-trajectories.mjs`, `neuralese-convert-trajectories.mjs` (data side only) |
| 3 Converted-task builds, both forms executed | not started | — |
| 4 Graph-level replay trainer | built (smoke) | `train/graph_replay.py`: whole-program replay of `natlang.replay-record/1` records with propagated writes (S5 §3.2 step 5; S4 teacher forcing keeps recorded values, so upstream training needs propagation), soft-value bank, trained/initial/zeroed/shuffled controls and G2 gate; records from `ts-host/src/neuralese/replay-records.ts` (`learningService({replayRecords})`); producers now include function-captured calls (`learning.ts` `Recorder.nested`). Closures beyond recorded producers not covered |
| 5 Fresh-rollout pipeline recorded as graphs | partial | `ts-host/src/neuralese/recording.ts`; no campaign driver |
| 6 Text versions of operators, `compose`; trajectory collection | partial | Text bodies plus `buildStandardLibrary`; Python `stdlib.py`; LFM2.5-350M (pilot-v4 heads) text stdlib artifact `stdlib-lfm25-350m-pilotv4-text-20261009-v1`; operator cases `neuralese-operator-cases-20261009-v1` (`data/operator_cases.py`, code-computed targets) and collector `ts-host/scripts/neuralese-collect-operator-records.mjs` (read, map, zip/split, laws); no Mellum stdlib, `compose` not collected |
| 7 Operator training as soft values | built (smoke, unqualified) | `graph_replay` trains the stdlib bodies: 60 steps on 144 records; held 72: trained 2.989 vs initial 3.460, zeroed 3.169, shuffled 3.224 (G2 passed); artifact `stdlib-lfm25-350m-pilotv4-trained-20261009-v1` (superseded: context-free encoding). The smoke's text init encoded descriptions bare through the port and the legacy RMS read transport; bodies are now initialised in context (DECISIONS 2026-10-10, `text_init.py`, `init_body`). Full-size run relaunched 2026-10-10 (DGX, `runs/neuralese-s5-operator-full-20261010-v1/run.sh`, unit in events.log): `stdlib-lfm25-350m-luna28793-incontext-20261010-v1` (all init gates bit-exact) on Pop's LFM closed-output heads (latent-sketch-v2, cutoff 8, foundation unqualified), normal call trajectory for combinators, all 1166 train + first 200 test cases in 28 shards, 2 collector+server workers, then graph_replay for 2 epochs at batch 8 with 4 interim held evals and the G2 controls; resumable per shard. The first launch's shards (pilot-v4, context-free bodies) were discarded. No live runtime check yet |
| 8 Law losses and law suite | partial | Law objectives in `learning.ts` and the server; combineIdentity and splitZip collected and trained in the S5 smoke (held: 1.94→1.62, 2.26→1.64); no law suite per dialect |
| 9–13 Curriculum, literal data, agreement eval, G2 harness, data products | not started | — |

## S6 soft skills and meta-learning (with LEARNING_CONTINUUM M0–M7)

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Soft-skill format, `.nz` skills, text-init arms | used (LFM) | `skills/registry.ts`, `soft-skill-decision.mjs`; text init is in context (`initBodyInContext`, gate recorded with each init) since 2026-10-10; `encode`/`embed` remain named diagnostic arms |
| System-prompt bank: text-init plus trained | partial | LFM text-init banks `system-prompt-bank-lfm2.5-350m-text-init-20261004-v1/v2` superseded (context-free encoding, 2026-10-10); the builder initialises pieces in context (a sample call, else a minimal call's system message); `train.decision --soft-prompts` and `trajectories --bank` train and save `system-prompts.nz`; no Maple/Mellum bank yet (trajectory runs on Maple ran without `--bank`, so no bank was saved) |
| M0 records/arms, M2 deltas, M4 adapters, M5 P/D projections | used (LFM) | `step-record.ts`, `deltas.ts`, `model/tiny_adapters.py`, `model/projections.py` |
| M3 memetic optimiser (#15) | partial | `memetic-decision.mjs` |
| Learned updater v0 | partial (LFM, unqualified) | 2026-10-09: `train/learned_updater.py` (U block + base skill + support view → write unroll → D delta, fit to recorded soft-gold deltas), `evaluate-soft-skills.mjs`; artifact `learned-updater-v0-lfm2.5-350m-20261009`. Held-out 4 families query quality: soft-init 0.607, updater 0.601, gradient 1 step 0.633, 8 steps 0.802 (does not beat matched-compute gradient; delta-space held-out cosine 0.12). Next: more step records, D stage-1 init, serve D |
| Gradient digests, query-trained updater, self-revision | not started | — |
| M6 reward-blind improver | partial | `improvement/blind-view.ts` (views only) |

## S7 RL

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Advantage-weighted `logLikelihood` objective | partial | `learning.ts`, `train/losses.py` |
| Rollout task record, environment contract, rollout driver, GRPO round (group advantages, sequence ratios, clipping, KL shaping) | used (LFM smoke) | 2026-10-09: `plans/neuralese/S7_ROLLOUT_CONTRACT.md`, `scripts/rl/rollout-episodes.mjs`, `scripts/rl/build-decision-episodes.mjs`, `rl/grpo.py`; smoke on LFM2.5-350M: SQL/scifact/contractnli 112 rollouts all reward 0 (no spread); yes/no decision episodes give spread (boolq [1,1,1,1,0,0]); one GRPO update applied (36 terms, adapter nz1_tlb7… → nz1_nv4c…), runs/s7-rl-smoke-20261009 on the HDD |
| Token-level ratios, stale-round handling, monitors, rubric runner, research/CSP/selection families in rollouts | not started | research scorer: invalid host research reference; TS-helper families untested on a clean dist |

## S8 target and release

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Target choice | used | Mellum decision (DECISIONS.md, 2026-10-09) |
| Artifact library: registry, metadata, dialect pinning, publication | partial | 2026-10-09: `training/neuralese_artifacts.json` + `training/artifact-manifests/` (`natlang_neuralese/artifacts.py`, `scripts/neuralese_artifacts.py`: register, register-run, verify, resolve, push/pull); recipe binding `{"artifact": ID}`; `train.decision --register-artifact`; 4 LFM artifacts registered. Not done: HF publication, Maple/Mellum artifacts |
| GGUF builds, model cards, upstreaming | not started (for the target) | — |

## Counts

Out of 44 audited deliverables (updated 2026-10-09 after the artifact work and an audit correction):

| Class | Count |
| --- | --- |
| used | 16 |
| not wired | 0 |
| partial | 14 |
| not started | 14 |

## Priorities: what blocks a complete Mellum Neuralese system first

| # | Item | Owner |
| --- | --- | --- |
| 1 | Mellum QAT conversion qualifies at λ=1 | Mellum session (running) |
| 2 | Mellum foundation warm-up, runtime check, recurrence via shared recipes (#45), with nested members (#44) | Mellum session; recipes via the C3 recipe consolidation by dgx-claude-7351f337 |
| 3 | Artifact registry plus a standard recipe/trainer include path (done 2026-10-09 except publication); next: backbone-specific dialect tags (decision), Mellum runs pass `--bank` and register their banks | backlog agent; dialect: owner |
| 4 | Mellum text-initialised artifacts: standard library (`buildStandardLibrary` against a Mellum server) and system-prompt bank (from a Mellum trajectory run with `--bank`, or the TS builder against a Mellum server) | backlog agent once a Mellum server runs |
| 5 | S5 multi-call graph replay (credit across chains of calls and closures); per-call replay over converted records exists in `trajectories.py` | built 2026-10-10 (`train/graph_replay.py`, smoke-gated); next: full-size run, Mellum records |
| 6 | Operator/combinator training samples and operator training (S5 items 6–7) with law terms | built 2026-10-10 (smoke, unqualified artifacts); next: scale cases, live runtime check |
| 7 | Reference server serving Mellum students (QAT-converted latents, read adapter) | Mellum session |
| 8 | llama.cpp/wasm for Mellum (MoE + ternary) and refreshing the browser build to the fork pin | batched-scoring/fork agent |
| 9 | S7 rollout task record, environment contract and rollout driver (smoke done 2026-10-09; next: monitors, multi-round on Mellum) | RL sub-agent |
| 10 | Learned updater v0 (S6 items 6–8) from improvement-step records (v0 built, unqualified; next: more data, D init) | RL sub-agent |

Canonical evaluator (`eval/self_feedback.py`, prefix metrics, `prepare_text_windows`): Pop. C2/C3 optimizer and
recipe consolidation: dgx-claude-7351f337. Do not duplicate either.
