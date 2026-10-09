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
| 4 Graph-level replay trainer | partial | Per-call replay: `serve/grad.py` GradSession, `train/execution.py`; no multi-call graph trainer |
| 5 Fresh-rollout pipeline recorded as graphs | partial | `ts-host/src/neuralese/recording.ts`; no campaign driver |
| 6 Text versions of operators, `compose`; trajectory collection | partial | Text bodies plus `buildStandardLibrary` in `ts-host/src/neuralese/combinators.ts`; no Mellum stdlib built, no collection |
| 7 Operator training as soft values | not started | — |
| 8 Law losses and law suite | partial | Law objectives in `learning.ts` and the server; no law measurements per dialect |
| 9–13 Curriculum, literal data, agreement eval, G2 harness, data products | not started | — |

## S6 soft skills and meta-learning (with LEARNING_CONTINUUM M0–M7)

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Soft-skill format, `.nz` skills, text-init arms | used (LFM) | `skills/registry.ts`, `soft-skill-decision.mjs` |
| System-prompt bank: text-init plus trained | partial | LFM text-init banks registered (`system-prompt-bank-lfm2.5-350m-text-init-20261004-v1/v2`); `train.decision --soft-prompts` and `trajectories --bank` train and save `system-prompts.nz`; no Maple/Mellum bank yet (trajectory runs on Maple ran without `--bank`, so no bank was saved) |
| M0 records/arms, M2 deltas, M4 adapters, M5 P/D projections | used (LFM) | `step-record.ts`, `deltas.ts`, `model/tiny_adapters.py`, `model/projections.py` |
| M3 memetic optimiser (#15) | partial | `memetic-decision.mjs` |
| Learned updater, gradient digests, query-trained updater, self-revision | not started | — |
| M6 reward-blind improver | partial | `improvement/blind-view.ts` (views only) |

## S7 RL

| Deliverable | Class | Evidence |
| --- | --- | --- |
| Advantage-weighted `logLikelihood` objective | partial | `learning.ts`, `train/losses.py` |
| Rollout task record, environment contract, rollout driver, importance ratios/clipping/KL, monitors | not started | — |

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
| 5 | S5 multi-call graph replay (credit across chains of calls and closures); per-call replay over converted records exists in `trajectories.py` | backlog agent, in new modules |
| 6 | Operator/combinator training samples and operator training (S5 items 6–7) with law terms | backlog agent |
| 7 | Reference server serving Mellum students (QAT-converted latents, read adapter) | Mellum session |
| 8 | llama.cpp/wasm for Mellum (MoE + ternary) and refreshing the browser build to the fork pin | batched-scoring/fork agent |
| 9 | S7 rollout task record, environment contract and rollout driver | unassigned |
| 10 | Learned updater v0 (S6 items 6–8) from improvement-step records | unassigned |

Canonical evaluator (`eval/self_feedback.py`, prefix metrics, `prepare_text_windows`): Pop. C2/C3 optimizer and
recipe consolidation: dgx-claude-7351f337. Do not duplicate either.
