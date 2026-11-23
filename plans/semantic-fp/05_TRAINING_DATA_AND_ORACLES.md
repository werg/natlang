# WS5: Train what semantic FP needs, through the expert-iteration loop that actually runs

**Parent:** [Campaign index](README.md). **v2** (2026-11-23). v1 targeted a GRPO-first pool with "target pool shares".
The method is **expert iteration** (owner-approved 2026-11-20). One loop draws tasks from the registry role
`grpo-pool` by frontier tier. It builds a per-task pool of verified trajectories: own best-of-k successes first, then
leak-checked `jolt-guided` successes, then teacher trajectories with student-filled reasoning. It trains LoRA SFT on
that pool and gates on held-v3 on the DGX series. GRPO is the polish phase. New constructs therefore need two
things: **tasks with real oracles** (so collection can find own successes and the frontier can send the rest to
teachers) and **teacher demonstrations of idiomatic programs** (so the teacher tier has something to fill). The
measurement comes first: held-v4 is built now, in Wave A.

## 1. Entry points (exact)

| Step | Entry point | Notes (verified 2026-11-23) |
|---|---|---|
| Convert authored tasks | `scripts/rl_tasks/sources/authored.py IN.jsonl OUT.jsonl` (`natlang.rl-authored-task/1` → `natlang.program/2`, `ts-host/src/teacher/program.ts:16`) | `ORACLES = {'exact','normalized'}` (`authored.py:42`). O1 extends it. |
| Static checks | `scripts/rl_tasks/authored_checks.py` | adds idiom floor (WS3 I1) and keyword-baseline check (WS3 T3) |
| Qualify | `scripts/rl_tasks/qualify_authored.py`, `verify.mjs RUNTIME IN.jsonl OUT [CONC]` (reference replay, negative controls, alias probe), `structure_smoke.mjs`, `blind_solve.py` (Sonnet, exact tasks only) | |
| Admit and assemble | `admit_authored.py`, `build_supply.py`, `measure_repetition.py`, `simulate_selection.py` | repetition measured before admission (AGENTS.md) |
| Register and promote | `scripts/data_registry.py record` / `promote ID --role grpo-pool` (`promote()` L176 → `register_corpora.register`); entries in `training/corpora/<id>.json` + `training/corpus-manifests/<id>.json` | 18 `grpo-pool` entries today, 89.7% from two early corpora |
| Selection | `scripts/neuralese_data/grpo.py` `default_pools()` = `role_paths('grpo-pool')` (L125); `load_tasks` (L3759) removes held closure and errata; `expert_iteration.py` `tier()`/`selection_order()` (L123/L133): sometimes → never → unmeasured → solved | the frontier sends never-solved tasks to teachers |
| Collection (own successes, JOLT+) | `expert_iteration.py collect` / `async-drive` (`--teacher-attempts`, default 2 at HEAD; `--neuralese-variants`) | JOLT+ has not yet produced expert-iteration data: the cycle that was running then was pinned before it was turned on |
| Teacher demonstrations | `expert_iteration.py` `TEACHERS` (L705: `claude-haiku-5-5`/anthropic, `gpt-6-luna`/openai-codex, `gpt-6.1-sol`/openai-codex, NVIDIA nemotron) → `teacher_gen.py --teacher JSON` (`ENDPOINT_PROVIDERS` L231: nvidia-nim, openrouter, google-gemini, zai, anthropic, openai); `GAP_TIERS` | free models enter as an `ENDPOINT_PROVIDERS` endpoint spec |
| Live teacher work | `vibestudio_campaign.py`, `live_campaign_ingest.py poll`, `sft_sources extract --live-run`, `async-drive --live-work WORK` | |
| Filled reasoning | `scripts/neuralese_data/filled_reasoning.py generate --work W [--prefer-work]` | teacher reasoning is never a target |
| Mix build | `sft_sources.py build` (`OWN_SOURCES = ('rl-own','jolt-guided')` L594, `FRONTIER_TIERS` L682, `frontier_drop()` L701, `THINK_TREATMENTS` L629; `--aux NAME=PATH,replaces=teacher`); `expert_iteration.py mix_build_command()` L786 with `TEACHER_POOLS` L701 | |
| Held | `rl/held_eval.py` `pin`, `verify`, `measure`, `compare`; `grpo.default_held()` = `role_path('held-eval')` (exactly one entry) | H1 changes this |
| Executable verifier (exists) | `trusted_exact_verifier: { path, sha256, oracle }` (`ts-host/src/teacher/trusted-exact-verifier.ts`): a `.mjs` under `training/semantic_applications/` exporting `verifyCandidate` | the only executable-oracle path today; O1 builds on it |

## 2. Oracles that make real semantics easier to verify than keyword rules (O1-O3)

v1's four oracle classes stand. v2 builds them on the existing mechanisms rather than as new modes:

- **O1, execution verifier** (2 days). Authored tasks may carry `trusted_exact_verifier` (`authored.py` accepts the
  `verifier` oracle; the verifier `.mjs` lives under `training/semantic_applications/` or a new
  `training/semantic_fp_tasks/` root added to the trusted-verifier allow-list in the same commit). The verifier
  returns graded `[0, 1]` + pass, as file-effect credit already does for `--reward graded`. Qualification: the
  reference passes, ≥ 2 negative solutions and the keyword baseline (WS3 T3) fail. Sources: migrated programs (WS7),
  flagship fixtures (WS6), `applications/{notebook,helpdesk,nldb,wiki,workflow,games}` fixture suites.
- **O2, metamorphic twins** (2 days). A task group holds paraphrase twins with low lexical overlap (content-lemma
  Jaccard < 0.25) and the same semantic outcome. A trajectory counts as a success when it passes its own twin. The
  *selection* pass rate of the group requires both, so a cue-word program cannot sit in the `sometimes` tier.
- **O3, proper scoring for decisions** (2 days). `readout: decision` tasks score Brier / bounded log-loss /
  asymmetric utility against the label. They re-enable floor-abstention tasks (`sources/decisions.py:20` excluded
  them). **SFT of decisions:** *(v2 fact)* an option-normalized cross-entropy already exists (`train/decision_supervision.py`,
  `common/decision_objective.py`), but only the native-graph path uses it. `exp0_sft` trains an own decision reply as
  plain token CE on the chosen option. O3 wires the existing objective into `exp0_sft` for decision rows, with targets
  = the verified label (own successes) or the teacher's scored option distribution where the teacher endpoint returns
  logprobs. This is first-principles (it trains the quantity `decide()` reads), so it ships as the default with
  `--no-decision-objective` as the opt-out.
- **Relational / provenance checks** need no new class: they are O1 verifiers that call `ensures:` checks (WS1 L7).

## 3. Task families for the constructs at 0-2% (T1)

No target shares. The frontier allocates collection. The supply must exist, and repetition must be measured before
admission (AGENTS.md: never pad scarce behaviours by repetition). Families, each with ≥ 20 distinct seeds before its
first admission:

| Family | Construct | Oracle | Seeds |
|---|---|---|---|
| `inline-lambda-pipeline` | inline `nl<T>` mapped over structured data in eval, `nl.with`, typed holes | O1 exact aggregation | decisions/reading datasets restated as collections; migrated briefs |
| `library-discovery` | find and compose helpers in a 10-30 item mixed `.ts`/`.nl` library with `read_code` / `search_code`; root prompt names no helper | O1 | migrated briefs and Astra apps (WS7), `applications/{notebook,helpdesk,nldb,wiki,logs,workflow,games}` |
| `semantic-convergence` | `iterateOn(...).until(nl\`...\`)`, `semanticRepair` against a real checker | O1 | compiler/test/schema errors from flagship fixtures |
| `higher-order-policy` | `compileRule` / `.nl` returning a lambda, run on hidden cases; `.in(context)` lens swaps | O1 | rule books (support policies, routing), `examples/review` lenses |
| `folder-transaction` | `propose` → check → `accept` or `branch`; adopt-call (WS1 L3) | O1 | codemod and doc-drift fixtures |
| `untrusted-contract` | `Untrusted<T>` with injection payloads; `Is<T,P>` / `ensures:` repair | O1 + O2 | inbox/PR fixtures |
| `structural-recursion` | recursive `.nl` with `measure:` over trees | O1 | nested specs, repo trees |
| `decision-calibrated` | flat and record decisions, floors, `escalate` | O3 | `sources/decisions.py` datasets, flagship routes |
| `refactor-to-semantic-fp` | a prose program to rewrite, then pass its cases | O1 (equivalence service, WS7 M1) | WS7 pairs |
| `improve-program` | improve a held-separated program from its support invocations | confirmation-split gain (WS8 N5) | WS7 programs |
| `neuralese-pipeline` | `semanticReduce(..., { carry: 'neuralese' })`, `view` over long documents | O1 | existing Neuralese variants re-expressed through std |

Authors: WS3 T3 (`applications/task-author`, natlang), with Haiku and free models as the fallback models of its
calls. The Python scripts in §1 stay the crisp mechanism.

## 4. Teacher demonstrations of idiomatic programs (D1)

Expert iteration's teacher tier fills tasks the student never solves. For new constructs most tasks start
never-solved, so teacher trajectories are the bootstrap:
- `teacher_gen.py` runs over the T1 families' `never` tier (the frontier already sends it). Luna and Sol go through
  the `openai-codex` provider, Haiku through `anthropic`, free models through `ENDPOINT_PROVIDERS` (nvidia-nim,
  openrouter, google-gemini, zai). Only verified successes enter.
- Student-filled reasoning (`filled_reasoning generate --prefer-work`) replaces teacher reasoning, as for every
  teacher chain.
- **Idiomatic program demonstrations at the author level.** A teacher writes whole programs (`.ts` + `.nl` + types +
  examples.json) from a brief, checked by `natlang check`, the idiom floor and the program's verifier. These are
  `natlang init`-style authoring trajectories. They train the backbone as the executor of an authoring program (the
  task-author and migrator calls), not as a coding agent.
- Prompts given to teachers carry the same front-door text as consumers (WS4): the generated AGENTS.md and the
  authoring skill. That way the demonstrations match what users are taught.

## 5. held-v4 first (H1), held-v3 alongside

held-v4 measures what the campaign wants, so it is built in Wave A, before the families it measures are admitted:

1. **Code (1 day).** `grpo.default_held()` requires exactly one `held-eval` entry (`role_path`, data_registry L772).
   Change: role `held-eval` stays held-v3 (the gate series). New role `held-eval-semantic-fp` names held-v4.
   `held_eval.exclusion_ids()` and `grpo.load_tasks` exclude the union over every manifest with a `held-eval*` role.
   `expert_iteration.select`'s hardcoded `exclude_manifest=[]` (L156) takes them too. `collect_exposure.py` reads its
   pools from the registry instead of the hand list. Tests in `tests/neuralese/test_held_eval.py` and
   `test_expert_iteration.py`.
2. **Set (2 days, parallel with code).** 256 tasks, stratified over the T1 families. Built by `held_eval pin` over a
   combined cases file from: WS7's reserved 20 design briefs and 2 Astra/Sol apps (their own cases), flagship
   fixture held splits (WS6), and T1 seeds reserved by split group *before* any T1 admission. Every held task's
   `split_group` and source group are excluded from training through `also_excludes`. The `held_eval verify`
   exposure check runs against every pool.
3. **Measure.** The current policy (`ops/current-policy.json`) and the base, on the DGX series (`dgx-held`
   machinery, 4 samples), as soon as the set exists. Later held measurements of each async round measure both sets.
   The round gate stays held-v3 until held-v4 has two measured rounds. After that the gate is the pair: a
   significant regression on either one halts the driver, as the failure rule already does for held-v3.

Held-v3 continues unchanged as the continuity series.

**As built (B1, 2026-11-23).** Every registry role named `held-eval*` is a held set training excludes:
`held_eval.training_exclusion_ids` (and `sft_sources.held_closure` for the SFT/teacher/live paths) unite them, so a
set is out of training the moment its role is registered. `held-eval` stays held-v3 (`grpo.default_held()`, the
gate). `held-eval-semantic-fp-reservations` is `ops/held-v4/reservations.json`
(`natlang.rl-held-reservations/1`: groups, path patterns and seeded share rules, `held_eval.reserved`), and
`held-eval-semantic-fp` will name the pinned manifest. Fact correction: a reserved brief's 8 cases are 3 exact gates
and 5 workflows whose oracle is an independent review rubric (`ts-host/scripts/design-briefs/lib.mjs` `makeRecord`).
So the exact stratum is 60 gates. The 100 workflows join held-v4 with the verifier oracle (O1/B6), as do the 2 reserved
Astra/Sol apps, whose fixtures are judged by verifier modules.

## 6. Activate fallow assets (F1)
- `training/semantic_stack/sol_programs.json` (100 typed programs, 28 worlds, "eligible for plain teacher
  generation", never admitted): run it through the WS7 migrator and the O1 path. Admit what qualifies.
- `applications/*` fixture suites → O1 tasks (`scripts/rl_tasks/sources/flagship_apps.py`).
- Memetic tasks: expert-iteration `collect` forces `--memetic-share 0`. Let collection draw `grpo-memetic-tasks` at
  the GRPO share (0.02) so improvement episodes' own successes reach SFT (WS8 N5 broadens the family).
- VibeStudio live corpus: the round driver's `--live-work` argument must be present on the running lineage (DECISIONS
  2026-11-22 22:10 "Driver restart needed"). The supervising agent checks this on the live driver; it is not a
  campaign item, but the campaign depends on it.

## 7. Acceptance for WS5
- held-v4 exists, is registered, is excluded from training by the code in H1, and has a base and current-policy
  measurement.
- Every T1 family has an admitted batch whose keyword baselines fail, measured for repetition and verified against
  held-v3 and held-v4.
- An async round's mix contains rows from the new families: own, jolt-guided or filled teacher. The build's report
  shows them.
- Decision rows train with the option-normalized objective by default.
