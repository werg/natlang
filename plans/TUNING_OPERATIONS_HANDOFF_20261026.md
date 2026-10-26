# Program tuning: operations handoff (2026-10-26)

For the agent that runs operations. The mechanism from
[HIERARCHICAL_LORA_TUNING.md](HIERARCHICAL_LORA_TUNING.md) is implemented on `main` (user docs:
[docs/TUNING.md](../docs/TUNING.md); decomposition: [applications/tuner/DECOMPOSITION.md](../applications/tuner/DECOMPOSITION.md)).
Everything below needs GPUs, an authorized teacher or both, so it was not run by the implementing session. The
implementing session verified mechanics only (one CPU step, real LFM2.5-350M, managed llama.cpp serving, strict
validation, activation). No capability claim exists yet: the end-to-end run tuned 0/2 vs base 0/2 at `steps=1`.

## What exists

| Piece | Where | Verified |
| --- | --- | --- |
| Scope tree, per-call stacks, release store, activation/rollback | `ts-host/src/adaptation/weights/` | unit tests |
| `natlang tune` (collect, split, train, export, validate, activate, report) | `ts-host/src/tune/`, `ts-host/src/cli/tune.ts` | unit + one real E2E (CPU) |
| Per-root incremental data, persisted split receipts | `ts-host/src/tune/{keys,collect,pipeline}.ts` | unit tests |
| Joint trainer (nested shared bases / free LoRA, soft instruction and skill states) | `training/neuralese/natlang_neuralese/tune/` | tests on real LFM2.5-350M weights |
| SFT and RL objectives, per-group learning rates, warmup+cosine, epochs, dev early stopping | `tune/trainer.py`, `tune/rl.py`, `ts-host/src/tune/{rollouts,rounds,schedule}.ts`, `natlang tune --rounds` | Python on real LFM weights; TS with scripted sampling receipts |
| Teacher calibration and routing | `natlang tune --calibrate`, `ts-host/src/tune/teachers.ts` | unit tests |
| Persistent inline-site identities, rename migrations | `ts-host/src/tune/sites.ts`, `tune.migrations.json` | unit tests |
| Luna-pool case export/import | `--export-cases`, `--import-cases`, `ts-host/src/tune/pool.ts` | real collector CLI with a scripted model |
| Cross-program families | `natlang tune --families <programs...>`, `tune/families.py`, `ts-host/src/tune/families.ts` | stand-in trainer and small random LFM fixture only |
| Distillation export for base SFT | `natlang tune --distill`, `tune/distill.py` | real LFM lowering test; stand-in stack |
| wllama (browser) per-request LoRA | `ts-host/vendor/wllama` (patched, `~/wllama-natlang` branch `natlang-lora`, unpushed) | headless Chromium on real LFM2.5-350M, 7/7 checks |
| Paired evaluation, placement, sweeps, report | `ts-host/src/tune/{evaluate,stats,placement,sweep,report}.ts` | unit tests |
| Managed llama.cpp serving (per-node GGUF, per-request scales) | upstream runtime pin v0.5.0 (b11146) | real `natlang call` on LFM2.5-350M |
| llama.cpp fork per-sequence LoRA (`--lora-per-sequence`) | `~/llama.cpp-neuralese` branch `neuralese` (unpushed) | LFM dense: bitwise CPU+CUDA; MoE expert path (tiny Qwen3-MoE): CPU tokens identical, |Δlogprob| ≤ 7e-6; CUDA MoE batches differ the same way without adapters |
| Native reference server (exact coefficients + soft states) | profile `neuraleseServer: true` | unit tests |
| vLLM whole-stack PEFT LoRAs | profile `vllm: true`, `tune/stack_export.py` | mock-vLLM test; no real vLLM run |

## Configuration

`natlang setup` writes, or edit `$NATLANG_CONFIG_HOME/config.json`:

```json
{ "tune": {
    "student": "lfm2.5-350m",
    "students": { "lfm2.5-350m": {
      "base": { "...": "exact native identity of the checkpoint below" },
      "checkpoint": "/path/native.pt",
      "engine": { "device": "cuda" },
      "trainer": ["/home/werg/natlang/.venv-neuralese/bin/python", "-m", "natlang_neuralese.tune.build"],
      "env": { "PYTHONPATH": "/home/werg/natlang/training/neuralese:/home/werg/natlang" },
      "hfConfigDir": "/path/hf-config-of-the-base",
      "baseGgufSha256": "sha256 of the GGUF the managed server serves",
      "moduleMap": "lfm2" } },
    "teacher": "<model profile name of the authorized teacher>",
    "backend": "managed-local",
    "recipe": { "steps": 400, "batch_rows": 16 } } }
```

Use `moduleMap: "mellum"` and Mellum's checkpoint/HF config for Mellum. The trainer runs on the configured device
from the native checkpoint; the served GGUF must be the same base (exports bind `baseGgufSha256`).

**Teacher.** Collection executes the program with the configured teacher profile; the trajectories become training
rows. Use only an owner-authorized teacher with exact actor provenance: the self-hosted Qwen3.6-35B-A3B generation
sandbox, or the six-worker Luna pool the owner resumed on 2026-10-26 (through its existing dispatcher, not a new
pool). The base student as teacher produced prose without tool calls and 0 admitted rows (correctly rejected).

## Runs to perform

1. **First real tune, LFM2.5-350M, GPU.** A program with ≥30 root examples with `expected` and `group`
   (candidates: the held-task packet shapes' training siblings, never the protected packet). `natlang tune ./prog`,
   then `natlang tune ./prog --evaluate development --with-teacher`. Record time-to-first-tuned-call from the report.
   Measured CPU cost: ≈1021 s per step (1 step, 350M) — use the GPU.
2. **Recipe sweep.** `natlang tune ./prog --sweep spec.json` (rank, penalty, data size, learning rates). It is
   resumable and journals per root; select on development, confirm once on `--evaluate confirmation`.
3. **Placement.** `natlang tune ./prog --placement`; check the selected stack's development loss stays within
   `tune.placement.maxSuccessLoss`.
4. **RL rounds.** `natlang tune ./prog --rounds` (settings `tune.rounds`: schedule default `sft,rl,rl,rl`,
   `samples` 4, `temperature` 0.8, `patience`). On the native reference server backend (`backend: "native"`, profile `neuraleseServer: true`),
   which records the native rollouts the RL objective needs. Start from the SFT release.
5. **Backend parity.** Same release served by managed llama.cpp, the fork with `--lora-per-sequence`, the native
   server and vLLM (`vllm-node:latest`, 0.29.1rc1 supports LFM2 LoRA); compare paired development success. Mellum's
   expert path in the fork is unrun.
6. **Mellum.** Repeat 1–3 with the Mellum student.
6a. **On-policy repair rounds.** Add `repair` rounds to `tune.rounds.schedule` (e.g. `["sft","repair","rl"]`) and run
   `natlang tune ./prog --rounds`. Each wrong training root costs one teacher run per tried boundary (latest first;
   `repairBoundaries` caps it). Report admitted/rejected repairs and rejection reasons from
   `round-<n>.repairs.jsonl`, and the development delta of the repair round. Works on every backend.
7. **Teacher calibration** (when more than one authorized teacher is configured in `tune.teachers`):
   `natlang tune ./prog --calibrate` before collection; record the calibration table with the run.
8. **Families.** After at least two programs are tuned on the same exact base: `natlang tune --families ./a ./b ...`.
   Report published versions, member gates and paired development deltas (negative transfer is flagged). Family
   recipe defaults (rank 8, 8 steps) are untuned.
9. **Distillation.** `natlang tune ./prog --distill` writes `.natlang/tune/distill/<release>/{turns.jsonl,
   native-graphs.jsonl,manifest.json,lowered.json}` (training split only, `admission: not-admitted`). Admission into
   a student's production SFT recipe is a normal registry decision (`training/neuralese_corpora.json`, immutable
   manifest); after the base is retrained, rerun `natlang tune` on the new exact base to rebuild every adapter.
10. **wllama.** Browser parity for one release (`scripts/wllama-lora-smoke.mjs` pattern); compat build and WebGPU
   unrun.

Report each run with: release digest, split receipt, row counts, recipe, paired success with clustered intervals,
and timings. Tuning a program does not train the production lineages and gives them no exposure credit.

## Not yet implemented (implementation work, not operations)

Open: a family schedule and member-vs-non-member transfer
reports, soft states on the llama.cpp fork's Neuralese server and in the browser, OPFS caching, upstreaming the
llama.cpp and wllama patches. Earlier list for reference:

Cross-program families (`--families`), base distillation, on-policy repair rounds, refitting only dirty nodes,
wllama LoRA blobs, soft states on the fork's Neuralese server.

## Incremental builds

`natlang tune ./prog --incremental` after an edit builds a candidate (edited leaves against pinned released
parents), then the shared refit of dirty ancestors; every `tune.incremental.fullRefitEvery` (default 5) builds, or with
`--full`, a full joint refit. For operations: after the first real tune, edit one definition, run `--incremental`, and
record the candidate's and the shared refit's paired development success and wall time against a `--full` refit of
the same edit (the measurement that decides whether candidates are worth activating, and the cadence default).

## Collecting tune data through the six-worker Luna pool

`natlang tune` never starts a pool. When the teacher should be the owner-resumed Luna pool, collection is a case
list that goes through the existing reviewed dispatcher (`scripts/run_reviewed_luna_dispatcher.py`), unchanged:

1. Export (no model calls): `natlang tune ./prog --export-cases runs/tune-cases/<prog>-<n>` writes `source.jsonl`
   (`natlang.program/2` rows with `source_groups`, `split`, and a `tune` binding), `cases.json` (index + seed) and
   `manifest.json` (source/cases sha256, build, scaffold and split digests, skipped roots). Only train/development
   roots with `expected` and without valid data are included (`--include-confirmation` to add confirmation).
2. Inventory: `/home/werg/natlang/.venv-neuralese/bin/python scripts/luna_source_inventory.py build --source
   runs/tune-cases/<prog>-<n>/source.jsonl --cases runs/tune-cases/<prog>-<n>/cases.json --output
   runs/tune-cases/<prog>-<n>/inventory.json` (verified on an export on 2026-10-26).
3. Root review: write a `natlang.reviewed_luna_dispatch_plan/2` for this source exactly as for other campaigns
   (`root_approved`, campaign root under `runs/`, frozen runtime receipt, authority, pinned `artifact_hashes`
   including the inventory; `slots` within the pool's six). Schedule it into the existing pool alongside or after the
   current campaign; do not start a second pool.
4. Import: `natlang tune ./prog --import-cases <campaign_root>` (walks `slot-*/results/*.results.jsonl`; other JSONL is
   ignored). It reports imported/rejected rows and turns without a binding. Rows from another program build, scope
   tree or split are rejected; re-export after edits.
5. Tune: `natlang tune ./prog` reuses the imported records (the `data` stage reports reused) and trains on rows
   whose `teacher` is the exact collector identity (`collector:<provider>:<model>:<hash of provenance>`).

Limits: programs with program guidance/imports/services are refused at export; inline-site invocations whose
virtual definition IDs do not match the build count as turns without a binding (named `.nl` definitions map
exactly); imported turns carry instruction texts for named definitions but not skill documents, so soft-skill states
get no imported rows. The collector's `GENERATION_GUIDANCE` is removed from system messages at import (verified
byte-equal to the serving runtime's rendering in `ts-host/test/tune-luna-bridge.test.mjs`).

