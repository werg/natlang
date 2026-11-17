# Program tuning and LoRA-bank meta-learning on the LFM3 stack (restore, 2026-11-17)

Owner, 2026-11-17: "the tuning / LoRA approach as well [is very important]." This note inventories the per-program
LoRA tuning and bank meta-learning mechanism as it stands on main (b746bbb19), says what works on LFM3-1.7B, what the
w5 retirement (60d2c1209 and 5500b44fb, 2026-11-12) broke and what is stale, and gives the smallest design that brings
both back on today's stack with task-outcome measurement. Earlier plans: `plans/HIERARCHICAL_LORA_TUNING.md`,
`plans/TUNING_OPERATIONS_HANDOFF_20261026.md`, `plans/PROGRAM_TUNING_RUNS_20261026.md`,
`plans/neuralese/CONTINUOUS_BANK_METALEARNING_20261103.md`.

Paths: `T/` = `training/neuralese/natlang_neuralese/`, `TS/` = `ts-host/src/`.

## 1. Inventory, end to end

### Fit (one program)

- **Fitter:** `python -m natlang_neuralese.tune.build --plan plan.json --out DIR` (`T/tune/build.py:167`): one
  resumable joint fit (`T/tune/trainer.py` `JointTrainer`, `Recipe`). Every row runs one native graph through
  `serve.native_grad` (the native engine, `serve.load_native_engine` -> `model/native_model.load_native_checkpoint`,
  which picks `lfm3_port` for LFM3, `native_model.py:234-265`). Adapter parameters: per scope node either a subspace
  core on a canonical SVD basis or free LoRA factors (`T/tune/parameterization.py`), FP32, gradients at FP32
  (`gradient_precision='fp32'`), optimizer state as native artifacts (`model/native_adapters.optimize_adapter`).
  The frozen base runs at the plan's `engine.dtype`.
- **Rows:** `natlang.tune-turn/v1` turns (`T/tune/rows.py`): exact request messages, tools and reply rendered with
  the student's serving template (`serve.targets.target_items`); reply is the target, prompt at `prompt_weight`
  (default 0 since 2026-10-27); each tool call bound to its action outcome (`natlang.tune-action-outcome/2`): failed
  or unverified actions are not targets, failed history stays as context.
- **Targets on LFM3:** `T/tune/targets.py:24-38` has an `lfm3` table (q/k/v/o_proj, the attention `gate_proj`,
  conv in/out_proj, feed_forward w1/w2/w3); `test_learned_optimizer_lfm3.py` checks it against the real port's
  inventory.
- **Product driver:** `natlang tune [PROGRAM]` (`TS/cli/tune.ts:81`, `TS/tune/pipeline.ts:425`): discover sites, split
  examples by connected component (~70/20/10 train/dev/confirmation, persisted `.natlang/tune/splits.json`), collect
  teacher (default: self-teaching) turns, write `natlang.tune-plan/v1`, run `tune.build`, seal the release
  (`TS/adaptation/weights/`), export, validate (tuned vs untuned arms on dev), activate. Rounds (SFT/RL/repair),
  sweeps, placement, incremental builds, families (`T/tune/families.py`), distillation (`T/tune/distill.py`),
  teacher calibration and the Luna pool import all exist (`plans/TUNING_OPERATIONS_HANDOFF_20261026.md` table).

### Bank and mixture labels

- **Bank** (`T/tune/learned/bank.py`): one FP32 orthonormal basis `U_m, V_m` (rank R) per target matrix, K
  components (low-rank cores in the basis, unit relative energy), 12 target groups on LFM2/LFM3 (depth quarter x
  matrix family), snapshot id `nbank1:<hash>`; saved as an `adapter-basis` artifact plus parameter sets.
- **Bank construction:** `tune.learned bank-init` (`__main__.py:90`; pooled gradient subspaces + weight SVD, one
  component per source from a short in-basis fit, `growth.py`), or `meta_bank.family_lora_fits` -> `delta_basis` ->
  `pca_components` (SVD of real per-family LoRA fits; `T/tune/learned/meta_bank.py`).
- **Meta-learning:** `meta_bank.BankMetaLearner` (Meta-SGD inner fit of signed per-group coefficients, 12 steps;
  first-order outer step into bank/basis/`a0`/`log_step`; weight-space regression to the PCA LoRA fits;
  `build_pool` holds out whole families: 3 support / 2 dev / >= 1 confirmation worlds per held family).
- **Mixture labels from actual fits:** `fit.fit_in_bank` (Adam on mixture logits and gains through the exact
  composition, `fit.py:140-180`) -> `targets.fit_labels`; free-tune targets from real `tune.build` fits on the bank
  basis (`targets.free_tune_targets`, `__main__.corpus`). `bank.fit_mixture` (weight-space least squares) is only
  initialization/growth.
- **Learned optimizer / mixture head:** `tune/learned/{mixture,writer,trainer,meta_rl,...}`, the controller program
  `applications/learned-optimizer/`, `natlang tune --learned SPEC` (`TS/tune/learned.ts`; `policy: crisp-baseline`
  runs `tune.learned.apply`, the meta-learned bank's inner fit).

### Apply / compose

- `bank.composed_cores` -> `compose_adapter` stores an ordinary native `xs` adapter slicing the bank basis
  (`bank-basis-slice-v1`); every native path runs it unchanged (`model/native_adapters.py:110,187`).
- Stacks: `T/tune/stack.py` sums a call's scope-chain members exactly (same basis adds in basis, else concatenates).
- Native application: `model/native_adapters.run_native` substitutes effective weights with torch.func (exact).

### Export

- PEFT safetensors (`T/tune/export.py:62` `export_peft`), GGUF LoRA (`export.py:121` via
  `vendor/llama.cpp/convert_lora_to_gguf.py`, which registers `Lfm3ForCausalLM`, `vendor/llama.cpp/conversion/lfm3.py:18`),
  per-stack PEFT for vLLM (`T/tune/stack_export.py --module-map lfm2|lfm3`), native `.nnt` artifacts (the release).
  `build.py:244,285-303` exports a GGUF per node when the plan asks for `exports: ['gguf']` with
  `student.hfConfigDir` and `student.baseGgufSha256`.

### Serving, per engine

| Engine | How an adapter is served | LFM3 state |
| --- | --- | --- |
| Managed llama-server (fork) | `--lora-init-without-apply --lora <gguf>` per adapter (sorted by id), `--lora-per-sequence` if the binary has it; per-request scales over all loaded adapters (`TS/model/lora-scales.ts:10`, `TS/model/local-server.ts:151`, `TS/llama/server.ts:238-264`); tuned runs share a server keyed by adapters/states (3484d627e) | converter registers LFM3; **no LFM3 GGUF LoRA has been exported or served**; `check_tuned_lora_llama_parity.py` ran on LFM2 only |
| In-page wllama | `TS/browser/tuned-release.ts:27` `loadBrowserRelease` (GGUF exports, hash and base checks), adapters at scale 0 + per-request scales (`TS/browser/local-model.ts:236-242,361`) | vendored natlang-b12081; **no LFM3 adapter run**; `ts-host/scripts/wllama-lora-smoke.mjs` manual |
| Python native server | release `.nnt` uploaded to the server store (`TS/tune/tuned-serving.ts:69-77,122`); requests carry `x_natlang_adapters: [{id, scale}]` (`TS/model/neuralese-server.ts:566-601`, `T/serve/http.py:550`, `T/serve/engine.py:408`) | works on LFM3 (`lfm3_port`); exact |
| SGLang actor (fork werg/sglang-natlang) | the front sums a request's native stack into rank-padded slot factors and sends them as `custom_params.natlang_adapter` (`T/serve/sglang_actor.py:161,373-470`; `--sglang-adapter-slots 4`, `--sglang-adapter-rank 32`); a stack over the slot rank or no slots falls back to the native engine. Not SGLang's own `--enable-lora` | LFM3 is the fork's model; parity checked by `scripts/neuralese_data/check_sglang_adapter_parity.py` (random xs/free-LoRA stacks) |
| vLLM (legacy) | per-stack PEFT via `POST /v1/load_lora_adapter` (`TS/tune/tuned-serving.ts:159`) | unverified; not one of the four engines; removal candidate |

`docs/ENGINE_PARITY.md` has no LoRA row (only "mixed adapters in one batch", :106); the LoRA backend matrix lives in
`docs/TUNING.md:286-302` and is LFM2-era.

### Outcome measurement

- Product: `TS/tune/evaluate.ts:88` `rootSuccess` (oracle score == 1), paired arms `untuned` / `hierarchy` with
  component-clustered intervals (`TS/tune/stats.ts`); rounds, sweeps, placement and families select on dev; confirmation
  only via `--evaluate confirmation`.
- Training side: `T/rl/held_eval.py` (`measure` against an endpoint whose `rl_policy_binding.checkpoint_sha256` is the
  policy id; `compare` with the within-task permutation p and task-bootstrap CI). It knows nothing about adapters.
- Deleted: `tune/learned/bank_outcomes.py` (held-family task-outcome gate: per family 4/4/4 support/dev/confirmation
  tasks from the RL pool; arms parent / candidate bank / published bank / LoRA@1/4/24 collected through
  `rl/production.collect` with `engine.native_default_adapters` set to the arm; promotion on a raw sum of successes,
  no significance test).

### CLI

`natlang tune` (TS) with `--examples`, `--evaluate`, `--rounds`, `--sweep`, `--placement`, `--incremental`, `--full`,
`--families`, `--held`, `--distill`, `--learned`, `--score-proposals`, `--export-cases`, `--import-cases`,
`--calibrate`. Python: `tune.build`, `tune.families discover|fit`, `tune.distill`, `tune.supply`, `tune.structure`,
`tune.stack_export`, `tune.learned <bank-init|targets|train|propose|ladder|proposals|service|collect|corpus|controls|meta-examples|adopt-bank|serve>`,
`tune.learned.apply`, `tune.learned.lane`.

## 2. What works, what broke, what is stale

**Works on LFM3 (by code; mostly untested on real weights):** every `tune/` and `tune/learned/` module imports (all 45
checked, AST-resolved); the native engine loads LFM3 and applies native adapters exactly; LFM3 target table and bank
groups (tested); SGLang adapter slots (parity script); GGUF converter registers LFM3; TS pipeline has no LFM2/Mellum
hard-coding (only comments and the test student's `moduleMap: 'lfm2'`).

**Broken by the deletion:**

- The bank meta-learning driver (`bank_stage.py`: pool, PCA init, meta-training loop, publish) and its task-outcome
  gate (`bank_outcomes.py`) are gone. `meta_bank.BankMetaLearner`, `build_pool`, `graph_rows`, `rollout_rows`,
  `family_lora_fits` and `promotion` survive but have no caller.
- `rl/production.collect` (the collector bank_outcomes scored with) is gone; `rl/task_rollouts` survives and
  `held_eval` is today's collector/scorer.
- The setter of `engine.native_default_adapters` (in `bank_outcomes`) is gone; the reader paths remain
  (`T/serve/engine.py:415`, `T/serve/sglang_actor.py:376`). Nothing can serve "base + this adapter for every request"
  to a harness that does not name adapters itself (held_eval's task episodes do not).
- The `program_adaptation` / `learned_optimizer` stages (`tune/stage.py`, `tune/learned/stage.py`), the recipe stage
  kinds (`train/adaptation_stages.py`, `train/recipe.py`, `train/promotion.py`) and their tests (5500b44fb:
  `test_adaptation_stages.py`, `test_bank_metalearning.py`).

**Never verified on LFM3:** a real `tune.build` fit on LFM3-1.7B; an LFM3 GGUF LoRA through llama-server or wllama;
LFM3 PEFT export; a bank on real LFM3 weights (`bank-init` with rank > 16 may trip on the 16-row attention gate:
`meta_bank.select_targets` drops short matrices, `__main__.bank_init` does not).

**Stale:**

- `tune.build` / `families` / `distill` / `learned.apply` default `engine.dtype` to `float32`, which on LFM3 is a
  diagnostic weights-only cast (`native_model.py:239-245,435-440`); LFM3 fits should run the frozen base in BF16 with
  FP32 adapter parameters (the AGENTS.md precision rule applied to the trainable part).
- `TS/tune/pipeline.ts:89` forwards only `progress:` lines; `build.py:239` prints JSON step lines (no progress shown).
- `natlang tune` activates a release without a tuned-vs-untuned gate (`pipeline.ts:471-476`); only rounds/placement/
  families gate on success.
- `docs/TUNING.md` backend matrix and parity evidence are LFM2.5-350M; `training/neuralese/learned-optimizer/README.md`,
  `dry_run_fixture.py`, `tune/learned/lane.py:2,205-228` and `tune/learned/DECOMPOSITION.md:24,36,53` cite deleted
  stages or `ts-host/src/tune/optimizer/` (now `applications/learned-optimizer/`).
- vLLM backend: not one of the four engines, LFM3 support unverified.

**Recorded results so far:** LFM2.5-350M ticket-routing dev 0/20 base -> 12/20 teacher-SFT adapter -> 17/20 after
SFT+RL rounds (10-27; no confirmation, no paired base at equal limits). Bank (11-03, PCA init, 64/16 families): at
step 0 the meta-bank's inner fit reached dev teacher-forced loss 0.675 vs parent 2.325 and LoRA@4 0.908; meta-training
stopped at outer step 5; no task outcome was ever measured.

## 3. Design: per-program LoRA and bank meta-learning on today's stack

Principles: reuse the one fitter (`tune.build` / `JointTrainer`), the native artifact store and the engines' existing
adapter paths; reuse `held_eval` for task outcomes; one small crisp driver for the training side (training stack:
standard tooling, minimal code). The product `natlang tune` stays the user-facing path over the same fitter.

### 3.1 Program, splits, data

- **Program** = one task family of the GRPO pools (`grpo.POOLS`, `natlang.program/2` rows: one typed function, many
  inputs). Candidates are families with many tasks and many verified successes (e.g. `ext_decisions_route`: 549
  tasks, 132 verified successes in `sft-sources-20261117`).
- **Splits:** pool rows of the family minus the held-v3/v2 closure (`held_eval.exclusion_ids`) and pool errata,
  grouped by `curriculum.split_group`, hash-ranked by `sha256(program-split:<family>:<group>)` into support (60%),
  dev (25%), confirmation (15%). Persisted as `split.json` with the pool digests; never recomputed.
- **Fit data:** verified successes of support tasks only (`sft_sources.py extract` + `replay` output: rl-own and
  teacher episodes replayed under the current runtime and oracle). Each call is one `natlang.tune-turn/v1` row (exact
  messages, tools, reply; prompt weight 0). Action evidence from the replay: a tool call whose result is a runtime error
  is `rejected` (not a target; kept as failed history), else `ok`; the episode's oracle verdict is the root evidence.
- **Exposure:** the start policy must not have trained on the program's dev/confirmation tasks. The untouched base
  5ff07638 has not; checkpoints trained on `sft-sources` builds or GRPO over the same pool have, so per-program splits
  on a trained parent exclude every task id in that parent's exposure (the run's selection records), or the program
  is measured on the base.

### 3.2 Fit, export

- New `scripts/neuralese_data/program_tune.py` (crisp driver) with `split`, `rows`, `fit`, `serve`, `measure`:
  `fit` writes a `tune.build` plan (one tree scope node, free LoRA rank 16 over the LFM3 table, Adam 2e-3,
  warmup+cosine, `engine.dtype bfloat16`, `exports: ['gguf']` with the base GGUF's sha and hf-config) and runs
  `tune.build` under the ledger (medium priority, <= 15 GB).
- Outputs: the release fragment (`result.json`, `artifacts/*.nnt`), the GGUF LoRA per node (`exports/`), the PEFT.

### 3.3 Serve on every engine

- **SGLang actor / Python native:** `serve_native_checkpoint.py --adapter-dir DIR --default-adapter ID[:SCALE]`
  imports the release artifacts into the front's store and sets `engine.native_default_adapters` (restoring the
  setter); the front's `rl_policy_binding` records `adapters` ([{id, scale}]) and the policy id of an adapted actor is
  `<checkpoint_sha256>+<adapter id>`. Requests that name adapters (`x_natlang_adapters`) keep working as today.
- **held_eval:** `measure` accepts an adapted policy id, records `adapters` in `run.json` and `compare` refuses to
  pair a result with itself across different adapter bindings unless they are the two arms being compared (the
  arms differ in policy only; engine, manifest, sampling stay identical).
- **llama-server (fork) and wllama:** the GGUF LoRA from 3.2 on the natlang-lfm3-5ff07638 Q8_0/F32 GGUF;
  `check_tuned_lora_llama_parity.py` generalised to LFM3 (attention + gate, conv, MLP targets) as the parity check
  against the native adapted forward; the page through `loadBrowserRelease` (manual smoke).

### 3.4 Measure

- A per-program held_eval manifest of the program's dev tasks (`held_eval` schema, pinned under the program's run
  directory with the pool digest), measured at 4 samples per task with the same engine pin as held-v3:
  base vs base+adapter, paired `compare` (mean per-task pass-rate change, permutation p, bootstrap CI).
  Confirmation tasks are measured once, for the selected release.
- Success: dev pass rate rises with p < 0.05. Failure: no gain at the declared fit budget -> diagnose (rows, masking,
  template parity between the fit's rendering and the actor's, served-adapter parity) before judging the method.

### 3.5 Bank meta-learning

- **Labels from actual fits:** the per-program fits of 3.2 over many families are the bank's material
  (`meta_bank.family_lora_fits` replaced by the stored `tune.build` fits: the "named bank" is built from those exact
  fits, and each fit's projection onto the bank is its label).
- **Driver:** `program_tune.py bank` restores the deleted `bank_stage` loop on the surviving `meta_bank`: pool from
  the programs' support rows (meta-train families: support/query from support tasks; held families: whole families
  withheld), PCA init from the fits, `BankMetaLearner.step` with teacher-forced dev loss as steering, publish
  `bank-manifest.json` with its `inner_fit` law.
- **Gate (task outcomes, replacing `bank_outcomes`):** for each held family, the bank's inner fit on that family's
  support rows composes one adapter (`compose_adapter`), served through 3.3; arms parent / bank inner fit / per-family
  LoRA, each a held_eval measurement on the family's dev manifest; promotion on the pooled paired compare across held
  families (bank vs parent, and bank vs LoRA at the same support data), p < 0.05; confirmation once.
- `natlang tune --learned` (crisp-baseline policy) then deploys a published bank to user programs unchanged.

### 3.6 Order and resources

1. Driver `split`/`rows`/`fit` + targeted tests (tiny engine fixture); one small real LFM3 fit under the ledger
   (medium, <= 15 GB, short) on one family.
2. Front `--default-adapter` + binding + held_eval adapted policies + tests; GGUF export and llama parity on LFM3.
3. Dev measurement base vs adapter (Pop when free of held evals; else a short DGX actor at medium priority).
4. Fits for >= 8 families, then the bank loop and its outcome gate.
5. DECISIONS.md and FALLOW_REGISTER rows at each milestone.

## 4. Running state (2026-11-17 evening)

- **Programs:** `/home/werg/data/program-tune-20261117/<family>/` for `ext_decisions_route` (support 315 / dev 155 /
  confirm 75), `ext_decisions_verdict` (315/122/80), `ext_decisions_tally` (241/100/54), `ext_bgkit_pubmed_reading`
  (718/332/184).
- **Support data from API teachers** (no actor free): `program_tune.py teacher-tasks` wrote
  `ops/teacher-gen-20261115/pool-progtune-decisions/` (160 support tasks per program, round-robin; dev/confirm never
  reach a teacher); `launch-multi.sh` there is pool-v2-rich's with only `--work` changed; ledger unit
  `tgen-progtune-2` (low, 7 GB). Then: `sft_sources.py extract --teacher WORK --out-name episodes-progtune`, `replay`,
  and `program_tune.py rows --dir P --sources WORK_SFT` per program, refit.
- **Factored LoRA in training** (`NATLANG_FACTORED_LORA`, default on, `0` opts out): a differentiated adapted
  operation runs each target's delta as `x A^T B^T` (FP32) beside the unchanged base projection
  (`native_adapters.FactoredLoRALinear` / `factored_lora_modules`, used by `run_native` and the invocation actors)
  instead of materializing `W + B A` and its FP32 gradient. Forward-only operations keep the exact substitution
  (inference cache, serving parity). Equal loss and coefficient gradients to the substitution on FP32 and BF16
  fixtures (test); the serving engines (SGLang slots, llama.cpp LoRA) apply adapters factored too.
- **Dev measurement on Pop** (`program-dev-pop.sh` in the programs root): one pin
  (`scripts/pin_checkout.py create COMMIT /home/werg/data/program-tune-20261117/pin-<c12>`), then
  `program-dev-pop.sh PIN P base 2`, `program-dev-pop.sh PIN P adapter 2`, `program-dev-pop.sh PIN P compare`.
  `pop_rl_actor.py start|serve-checkpoint --adapter-release DIR` copies the fit to Pop and serves it on one SGLang
  adapter slot of its rank; attestation and the reuse check include the release's result.json digest.

- **Support rows (2026-11-17 21:30):** teacher pool 636/640 tasks (4 provider-deferred, stopped), 373 successes; replay
  369 exact (after the decision-readout replay fix, 89ddedce7); `program_tune.py rows` (068085501: nested calls'
  last turns count as final) -> route 480 rows / 147 tasks, verdict 352 / 51, tally 1,182 / 121, bgkit 955 / 60.
- **Fits ready, not run:** `fit/plan.json` per program (all targets, rank 16, lr 2e-3, 60 steps x 8 rows);
  `fit-all.sh [PROGRAM ...]` in the programs root runs them one ledger unit at a time (medium, 15 GB) in a GPU window.
  The attention+conv exact-path fit3 (18/30 steps) is retired under `retired/`.

## 5. Not checked in this inventory

Nothing was run beyond imports: no pytest, no model load, no CLI `--help`. Not read in detail: `tune/trainer.py`
internals, `tune/learned/{trainer,writer,episode,views}.py`, the SGLang fork's adapter code, the llama.cpp fork's
`--lora-per-sequence` patch, `TS/tune/{incremental,pool,repairs}.ts`, the 11-03 bank run's logs beyond
`state.json`, and whether llama.cpp's LoRA loader applies LFM3's `o_proj` and attention gate in practice.
