# Program tuning: operations handoff (2026-10-26)

For the agent that runs operations. The mechanism from
[HIERARCHICAL_LORA_TUNING.md](HIERARCHICAL_LORA_TUNING.md) is implemented on `main` (user docs:
[docs/TUNING.md](../docs/TUNING.md); decomposition: [applications/tuner/DECOMPOSITION.md](../applications/tuner/DECOMPOSITION.md)).
Everything under "Runs to perform" needs GPUs, an authorized teacher or both; the implementing session ran none of
it. It verified mechanics only: tests with scripted models and stand-in trainers, Python tests on the real
LFM2.5-350M weights, and one real end-to-end CPU tune (one step, managed llama.cpp serving, strict validation,
activation). **No capability claim exists yet**: that run tuned 0/2 vs base 0/2 at `steps=1`. Tuning a program does
not train the production lineages and earns them no exposure credit.

## What exists

| Piece | Where | Verified |
| --- | --- | --- |
| Scope tree, per-call stacks, release store, activation/rollback | `ts-host/src/adaptation/weights/` | unit tests |
| `natlang tune` (collect, split, train, export, validate, activate, report) | `ts-host/src/tune/`, `ts-host/src/cli/tune.ts` | unit + one real E2E (CPU) |
| Per-root incremental data, persisted split receipts | `ts-host/src/tune/{keys,collect,pipeline}.ts` | unit tests |
| Persistent inline-site identities, rename migrations | `ts-host/src/tune/sites.ts`, `tune.migrations.json` | unit tests |
| Incremental builds: candidate (pinned parents, warm starts), shared refit, periodic full refit | `--incremental`, `--full`, `ts-host/src/tune/incremental.ts`, trainer pins/warm starts | Python on fixtures (frozen bit-identical, moments continue/restart honestly); TS with a stand-in trainer |
| Joint trainer (nested shared bases / free LoRA, soft instruction and skill states) | `training/neuralese/natlang_neuralese/tune/` | tests on real LFM2.5-350M weights |
| SFT and RL objectives, per-group learning rates, warmup+cosine, epochs, dev early stopping | `tune/trainer.py`, `tune/rl.py`, `ts-host/src/tune/{rollouts,rounds,schedule}.ts`, `--rounds` | Python on real LFM weights; TS with scripted sampling receipts |
| On-policy repair rounds (§4.4) | round kind `repair`, `ts-host/src/tune/repairs.ts`, `tune/rows.py` | scripted student/teacher; real-LFM row lowering |
| Captured service worlds (journal + replay for repairs) | `TuneOptions.services` (SDK hosts), `ts-host/src/tune/world.ts` | unit tests; repairs with a scripted service program |
| Teacher calibration and routing | `--calibrate`, `ts-host/src/tune/teachers.ts` | unit tests |
| Luna-pool case export/import | `--export-cases`, `--import-cases`, `ts-host/src/tune/pool.ts` | real collector CLI with a scripted model |
| Cross-program families: discovery, fit, publication, member refits, zero-shot transfer to non-members and `--held` programs, registry schedule (`tune.families.every`) | `--families`, `tune/families.py`, `ts-host/src/tune/families.ts` | stand-in trainer and small random LFM fixture only |
| Distillation export for base SFT | `--distill`, `tune/distill.py` | real LFM lowering test; stand-in stack |
| Paired evaluation, placement, sweeps, report | `ts-host/src/tune/{evaluate,stats,placement,sweep,report}.ts` | unit tests |
| Managed llama.cpp serving (per-node GGUF, per-request scales) | upstream runtime pin v0.5.0 (b11146) | real `natlang call` on LFM2.5-350M |
| llama.cpp fork per-sequence LoRA (`--lora-per-sequence`) | `~/llama.cpp-neuralese` branch `neuralese` (unpushed) | LFM dense: bitwise CPU+CUDA. MoE expert path (tiny random Qwen3-MoE, same `build_lora_mm_id` code Mellum uses): CPU tokens identical, \|Δlogprob\| ≤ 7e-6; CUDA MoE batches differ the same way without adapters |
| llama.cpp fork soft states (`--native-state`, `llama_state_span_register`) | same fork, commit `6eff8834f`; `tune/soft_export.py`; `tune.softStates: true` | real LFM2.5-350M on CPU: max \|Δlogit\| 0.043 vs the Python splice (plain-text baseline 0.075), top-1 1.0, perturbation control 2.68, plain-text logits bitwise unchanged; GPU smoke only |
| Native reference server (exact coefficients + soft states) | profile `neuraleseServer: true` | unit tests |
| vLLM whole-stack PEFT LoRAs | profile `vllm: true`, `tune/stack_export.py` | mock-vLLM test; no real vLLM run |
| wllama (browser) per-request LoRA, OPFS adapter cache | `ts-host/vendor/wllama` (patched wllama 3.12.0; source https://github.com/werg/wllama master), `loadBrowserRelease` | headless Chromium on real LFM2.5-350M, 7/7 checks (rerun after the upstream merge); cache with an OPFS stand-in |

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
    "teachers": ["<optional further authorized profiles, for --calibrate>"],
    "backend": "managed-local",
    "softStates": false,
    "recipe": { "steps": 400, "batch_rows": 16 },
    "rounds": { "schedule": ["sft", "repair", "rl"], "samples": 4, "temperature": 0.8, "patience": 1 },
    "incremental": { "fullRefitEvery": 5 },
    "families": { "every": 0 } } }
```

Use `moduleMap: "mellum"` and Mellum's checkpoint/HF config for Mellum. The trainer runs on the configured device
from the native checkpoint; the served GGUF must be the same base (exports bind `baseGgufSha256`). Backends:
`managed-local` (llama.cpp; the fork's `llama-server` adds per-sequence batching and, with `softStates: true`, soft
states), `native` (reference server; required for RL rounds), `vllm`.

**Teacher.** Collection executes the program with the configured teacher profile; the trajectories become training
rows. Use only an owner-authorized teacher with exact actor provenance: the self-hosted Qwen3.6-35B-A3B generation
sandbox, or the six-worker Luna pool the owner resumed on 2026-10-26 (through its existing dispatcher, see below; not
a new pool). The base student as teacher produced prose without tool calls and 0 admitted rows (correctly rejected).

**Services.** The CLI runs programs without host services. A service-using program is tuned from an SDK host that
passes `services: example => ({ ... })` in the tune options: fresh, test or sandbox services per root, never
production systems, because collection, evaluation and rollouts call them.

## Runs to perform

1. **First real tune, LFM2.5-350M, GPU.** A program with ≥30 root examples with `expected` and `group`
   (candidates: the held-task packet shapes' training siblings, never the protected packet). `natlang tune ./prog`,
   then `natlang tune ./prog --evaluate development --with-teacher`. Record time-to-first-tuned-call from the report.
   Measured CPU cost: ≈1021 s per step (1 step, 350M); use the GPU.
2. **Recipe sweep.** `natlang tune ./prog --sweep spec.json` (rank, penalty, data size, learning rates). It is
   resumable and journals per root; select on development, confirm once on `--evaluate confirmation`.
3. **Placement.** `natlang tune ./prog --placement`; check the selected stack's development loss stays within
   `tune.placement.maxSuccessLoss`.
4. **Rounds.** `natlang tune ./prog --rounds` with `tune.rounds.schedule`.
   - `rl` rounds need `backend: "native"` (a `neuraleseServer: true` profile), which records the sampling receipts.
   - `repair` rounds (on-policy corrections) work on every backend. Each wrong training root costs one teacher run per
     tried boundary (latest first; `repairBoundaries` caps it). Report admitted/rejected repairs and reasons from
     `round-<n>.repairs.jsonl` and each round's development delta.
5. **Incremental builds.** After the first real tune, edit one definition, run `--incremental`, and record the
   candidate's and the shared refit's paired development success and wall time against a `--full` refit of the same
   edit. This decides whether candidates are worth activating and the `fullRefitEvery` default. Note: with the
   default ranks the program root has a residual, so any edit also produces a shared refit.
6. **Backend parity.** Same release served by managed llama.cpp (upstream), the fork with `--lora-per-sequence`,
   the native server and vLLM (`vllm-node:latest`, 0.29.1rc1 supports LFM2 LoRA); compare paired development success.
7. **Soft states on the fork.** Build `build-cuda`/`build-cpu` from fork commit `6eff8834f`, point the managed runtime
   at its `llama-server` (`NATLANG_LLAMA_SERVER`), set `tune.softStates: true` with `backend: "managed-local"`, tune,
   and run `scripts/llama_state_span_parity.py <workdir>` once on the GPU build (verified on CPU only). LFM2 only.
8. **Mellum.** Repeat 1–3 with the Mellum student. Mellum soft states are not exportable to the fork yet (native
   backend only).
9. **Teacher calibration** (when `tune.teachers` lists more than one authorized profile): `natlang tune ./prog
   --calibrate` before collection; record the calibration table with the run.
10. **Families.** After at least two programs are tuned on the same exact base: `natlang tune --families ./a ./b ...
   [--held ./c]` (without arguments: every recorded program of the base). Report published versions, member gates,
   member deltas (negative transfer flagged) and the zero-shot transfer table for non-members and held programs.
   Family recipe defaults (rank 8, 8 steps) are untuned. `tune.families.every: N` runs this automatically after N
   newly activated program releases.
11. **Distillation.** `natlang tune ./prog --distill` writes `.natlang/tune/distill/<release>/{turns.jsonl,
   native-graphs.jsonl,manifest.json,lowered.json}` (training split only, `admission: not-admitted`). Admission into
   a student's production SFT recipe is a normal registry decision (`training/neuralese_corpora.json`, immutable
   manifest); after the base is retrained, rerun `natlang tune` on the new exact base to rebuild every adapter.
12. **wllama.** Browser parity for one release (`ts-host/scripts/wllama-lora-smoke.mjs` pattern); the compat build
   and WebGPU are unrun.

Report each run with: release digest and build kind, split receipt, row counts, recipe, paired success with
clustered intervals, and timings.

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

## Not yet implemented (implementation work, not operations)

- Mellum soft states on the llama.cpp fork; soft states in the browser (wllama).
- Batching mixed stacks in wllama (requests are serialized per stack).
- Repairs and rollouts for programs that use decision readouts (their decisions are not replayed or captured);
  native Neuralese graph provenance of a repaired student prefix; the repair boundary order is a fixed rule, not yet
  a pluggable policy.
- Natural-language policies (`routeTeacher.nl`, `proposeFamilies.nl`) are off by default and unmeasured
  (`plans/MODEL_FACING_CHANGES.md`).
- Upstreaming the llama.cpp and wllama patches to their original projects is an owner decision (external
  publication). Our copies are on GitHub: https://github.com/werg/llama.cpp-neuralese (branch `neuralese`, the fork's
  default) and https://github.com/werg/wllama (master; its llama.cpp submodule commit is the tag
  `wllama-natlang-lora-5aeac37` in werg/llama.cpp-neuralese).
