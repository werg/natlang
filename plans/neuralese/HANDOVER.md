# Neuralese programme: handover, 2026-10-03

State at the end of the first implementation session. Read [README.md](README.md) (decisions and stage graph) and [DECISIONS.md](DECISIONS.md) first; the stage plans S0–S8 in this directory are current. The source design documents are in [sources/](sources/) and are inputs, not the spec.

## 1. Owner working rules

- Plans and code live in this repository; commit to `main` (no push unless asked).
- Data from bgkit (`~/bgkit`) and Schnitzeljagd (`~/sdkb`) is reused; their models and machinery are not.
- No fixed pass/fail thresholds: evaluate continuously against rubrics, review, react.
- No budget or cost talk in plans; paid teachers are allowed.
- Dialects are a lightweight version tag; bump freely while the system is young.
- This machine **is the DGX Spark** (`mltick`, aarch64 GB10). It runs the Qwen3.6 vLLM teacher (`natlang-qwen36-nvfp4-server`, host port 8082) and a teacher campaign (user service `natlang-qwen36-workflow-v9-family-buffer1024-20261003`). **Do not stop them yourself; notify the owner when work is ready to take the whole machine.**
- Another agent (on the owner's dev machine) is fixing the ts-host test environment (dataset cache `vendor/datasets`, ffmpeg, `@natlang/node` staging, `bonsai-queue` hanging on `127.0.0.1:8081`, `llama-runtime` reading `~/.config/natlang`, `source-review`). Its work arrives on `main` as merges.

## 2. What is on main

| Stage | Commits (main) | State |
| --- | --- | --- |
| Plans | `c78800b43`, `352c10ee4` | S0–S8 plans, README, decisions; payload temperature added. |
| **S0 spec** | `41291918d` | `spec/SPEC.md` 0.5-draft (Contexts, Captures, Iteration and termination, Neuralese chapter), `TYPES.md`, `spec/neuralese.d.ts`, `spec/NEURALESE_{FILES,PORT,REWRITES,DIALECTS,GRAPH,DATA}.md`, schemas, `training/api-migrations/neuralese-language.json`, `conformance/neuralese/` (58 cases, many now implemented). Done. |
| **S1 data** | `95ec9ae78`, `96e3ac427`, `36c8c6873` | `spec/neuralese-port-record.schema.json`; ledger `training/neuralese_data_ledger.json`; converters `scripts/neuralese_data/`, CLI `scripts/neuralese_port_records.py`; dedup and split closure; multi-turn/agent converters; v13 migration driver skeleton `ts-host/scripts/migrate-neuralese-language.mjs`. |
| **S2 skills** | `31c4b6442` | `ts-host/src/skills/` (standard `SKILL.md` + `metadata.natlang`, pool, scope injection, disclosure), `spec/skill-episode.schema.json`, `ts-host/scripts/skills/` episode and repair-episode builders (samples only). |
| **S3 port** | `69c181586`, `0d3bef3e3`, `c3d58d435`, `5f2f3de00`, `0b5a68506`, `8f9866c47` | `training/neuralese/natlang_neuralese/`: own LFM2 forward with snapshot caches (fast path: SDPA GQA, preallocated KV, hub causal-conv1d), write/read ports, temperature-gated payload (μ, log σ, τ, KL, log-prob), phases A–F, batched ragged training, anti-collapse terms, harness, pilot driver `train/pilot.py`. |
| **S4 runtime** | `c2a287e7f`, `ee0f8655f`, `a9f979523`, `aa8ad73ac`, `1377c09eb`, `2913d9059` | ts-host: `Neuralese` type kind and diagnostics, literals ↔ references, content parts, in-memory store, iterateOn bounds and predicate prompt; contexts and rebinding, `.nz` files, `nl.with` lowering, companion `.nz`/skills binding; data rewrite passes + CLI `ts-host/scripts/neuralese-rewrite-trajectories.mjs`; execution graph `src/native/graph.ts`; law rewrite pass `src/compiler/rewrites.ts` (gated, no measurements yet); `natlang:learning` and `natlang:neuralese` (combinators as soft system functions). |
| **S4 server** | `49eb9bb1b`, `1af78e783`, `cda793803`, `9a33ef65a` | Python reference server `natlang_neuralese/serve/` (write procedure in decoding, store and block endpoints, escaping, batched steps, streaming, gradient replay sessions, optimiser steps); ts-host transport `src/model/neuralese-server.ts`. |
| **S4 llama.cpp** | `7d2522048` | Fork pin `training/neuralese/llama-cpp-fork.json`, GGUF export `natlang_neuralese/export/`, CPU fp32 parity tests (pass). |

Environment: Python venv `/home/werg/natlang/.venv-neuralese` (torch 2.11 cu130, transformers 5.18, peft 0.21; setup in `training/neuralese/README.md`). Tests: `.venv-neuralese/bin/python -m pytest -q tests/neuralese tests/test_neuralese_port_records.py`; ts-host: `npm run build` then `node --test test/neuralese*.test.mjs test/skills.test.mjs`. Beware the repository-wide `.gitignore` rule `data/` (a negation exists for `training/neuralese/natlang_neuralese/data/`); check `git status --ignored` before committing new code directories.

## 3. Unfinished work and where it is

| Item | Location | State |
| --- | --- | --- |
| llama.cpp server, GPU parity, heads on backend | Fork `/home/werg/llama.cpp-neuralese`, branch `neuralese`: `8bd8c95a0` (layer-range API, CPU read/write ports; parity passes) + `27b2e6e8b` **WIP** (neuralese-server.cpp, backend heads, GPU parity; unverified). Natlang side WIP on branch `worktree-agent-aa373bf2a5aa6f2f5` (`b94b43fe3`: GPU parity test variant, export CLI). | Resume: CUDA build for sm_121, GPU parity, heads on backend, llama-server protocol matching the Python server, ts-host end-to-end test, then wllama + OPFS. |
| S1 full conversion | `/mnt/external/natlang-development-data/data/neuralese/port-records/full-20261003/` (`raw/` ≈ 35 GB, `build.log`) | **Stopped during the SWE stage**; bgkit, Schnitzeljagd single-turn and multi-turn stages finished their raw output, but dedup, split closure across everything, validation and the manifest have not run. Resume with the build script from `36c8c6873` (see `build.log` for the stage commands). Licences for knights/synlogic: check the ledger. |
| S3 pilot | `runs/neuralese-s3-pilot-20261003-v2/` (`harness_after_C.json`, `metrics.jsonl`, `pilot.log`, `checkpoint.pt`) | After phase C the channel carried **no content** on held-out spans (correct 3.578 vs shuffled 3.568 nats, zeroed 4.888; cross-source similarity 0.99; effective rank 45.7). During phase D training batches showed correct beating shuffled by ~0.30 nats/token and similarity falling to 0.86, but the run was stopped before the held-out harness after D. The full-machine run plan was not written. |
| v13 corpus migration | `ts-host/scripts/migrate-neuralese-language.mjs` (skeleton) + `ts-host/scripts/neuralese-rewrite-trajectories.mjs` (passes) | Wire the passes into the driver and run over the v13 model-neutral corpus. |
| S2 corpus | `ts-host/scripts/skills/`, samples in `/mnt/external/natlang-development-data/data/neuralese/s2-skill-episodes-sample/` | No teacher collection yet. |

## 4. Next steps, in order

1. **S3 channel use.** Resume the pilot through phase D and read the held-out harness. If correct ≈ shuffled persists, strengthen content pressure (withheld sources, shuffled-negative contrast, self-distillation from the full-source teacher) before scaling. Then write the full-machine run plan and **ask the owner to pause the teacher campaign**.
2. **S1.** Finish the full conversion (SWE stage, dedup, closure, validate, manifest); run the v13 migration.
3. **llama.cpp.** Verify and finish the WIP commit; GPU parity; server; browser.
4. **S4 gaps.** Exact second-order `grad` (currently refused), `law` objectives, graph nodes for combinators/readouts asserted in tests, batched prefill/completion/readback in the server, `crossEntropy` target placement.
5. **S2.** Teacher collection of authoring episodes (paid teachers allowed; the DGX teacher is busy).
6. **vLLM** extension (S4 §7) for S7 rollouts.
7. Re-run the full ts-host suite once the environment agent's fixes land.

## 5. Worktrees

Agent worktrees are under `.claude/worktrees/` (excluded locally via `.git/info/exclude`). All finished work is cherry-picked to `main`; only `worktree-agent-aa373bf2a5aa6f2f5` holds unmerged WIP. The others can be removed with `git worktree remove` once you no longer need them.
