# Shared learner primitives: custody, collector, acceptance gate

The RL stage, JOLT+ teacher, memetic learner, tuning lane, metalearner and RL curriculum each need the same three things:

1. which worlds may teach, select or confirm;
2. how an episode is collected and recorded;
3. how an artifact is adopted.

Each has exactly one implementation, and every learner uses it. This document records each part's form and why it has that form. The natlang-native rule is "policy is natural language, mechanism is crisp". Everything here is mechanism: exact verifiers of what the stages produce, plus plumbing. The policies that use these mechanisms stay where they are:

- the allocation of components to slices;
- curriculum task selection (`selectTrainingTasks`);
- privileged fact selection (`selectPrivilegedFacts`);
- guidance authoring.

## 1. World-split custody: `common/custody.py`, `training/protected-worlds.json`

| Part | Form | Why |
| --- | --- | --- |
| `world_identities(case)` | crisp function | An exact equivalence key, so a model must not decide it. It is the union of every relation the earlier implementations checked: ids and source ids, source/split/pair groups (JOLT custody, held-collision checker); world bindings for evidence items, service worlds and inputs (memetic batch split); the rendered-world fingerprint (JOLT); and the visible-world key (held replenishment, curriculum dedupe, held-collision checker). |
| Single evidence items | identities for splitting only | Protected-world custody leaves single shared evidence items out (`evidence_items=False`), for two reasons. First, procedural generators draw them from shared banks ("no sun", "heater works"). Second, on the 20,640-row LFM3 RL curriculum, these items alone related 7,845 rows (38%) to held packets whose worlds differ. A world's whole evidence set still enters its rendered-world fingerprint. Memetic slices keep relating invocations by single items, as before. |
| `visible_world_key`, `world_fingerprint`, `world_bindings` | crisp helpers | Each is the single definition its former owners imported. |
| `ProtectedWorlds.load()` | crisp loader over a declared registry | It fails closed: an unreadable source, a source whose pinned digest differs, or no evaluation packets at all is a `CustodyError`. The registry lists every protected source: registered evaluation-only packets, held and quality source-group projections, decision holds and explicit test reservations. Before, each caller held its own partial list. |
| `ProtectedWorlds.check(case)` | crisp verifier | Returns the stable `natlang.jolt-privileged-custody/1` receipt (recorded in captures and rechecked by the learner), or raises `CustodyError`. Only `split == 'train'` worlds unrelated to every protected identity pass. |
| `related_components(rows)` | crisp union-find | Related invocations stay on one side. The split *policy* (memetic balance allocation, tuning hash draw, metalearner family split) assigns whole components. |
| `verify_slices` / `check_slice_receipt` | crisp verifier and receipt | Slices are pairwise unrelated and protected-clean. The receipt binds each slice's ids and row hashes to the protected digest, so an evaluator sandbox without the protected sources can re-verify it exactly. |

Callers:

- JOLT+ (`rl/jolt.teacher_entries`, `rl/native_stage.jolt_privileged_source`, `rl/privileged_context.case_facts`);
- memetic (`rl/memetic_batch.select` writes `custody.json`, which `rl/memetic_program.load_slices` verifies);
- the RL curriculum (`assemble_rl_train_curriculum.py` drops protected-related groups whole; `qualify_rl_train_curriculum.py` proves `protected_worlds: []`, which `rl/task_source` accepts);
- decontamination (`check_held_world_collisions.py`, `replenish_luna_held_shapes.world`).

## 2. One collector and rollout record: `rl/production.collect`

| Part | Form | Why |
| --- | --- | --- |
| `collect(..., system_prompt=None)` | crisp fan-out over the frozen collector (`scripts/run_bonsai_queue.py`) | It is the one path with actor-receipt attestation, owned worker sessions, retained native artifacts, block bundles and JOLT+ teacher attempts. Guidance under measurement is a `system_prompt` bound by digest into the collection binding (`guided: true`). Unguided collections keep their recorded binding. |
| `natlang.rollout/1` turns (`task_rollouts.rollout_turns`) | crisp record | Each turn now also carries the server's `native_state_imports` receipts, so "the stored state reached the executing model" is checked on the same record RL trains on. |
| `memetic_program.rollouts_reached` | crisp verifier over production rollouts | This replaced the memetic learner's own result-file walker (`captured_invocations`, `student_capture_reached`, `native_import_reached`) and its own shard runner (`memetic_task_runner.evaluate`). |

The tuning lane already enters through `tune/rl.py` → `rl/native_replay.prepare_objective` on `natlang.tune-rollout/v1` records. Those records wrap the same production episode.

## 3. One acceptance gate: `common/acceptance.py`

| Part | Form | Why |
| --- | --- | --- |
| `select(parent, candidate)` | crisp comparison | Development only. Paired seeds and limits, the same scored count, strictly more successes. Author or model judgement never decides adoption. |
| `require_untouched(confirmation, artifact, ledger)` | crisp ledger check | Runs before measuring, so spent confirmation worlds are never consulted for a different artifact. |
| `confirm(...)` | crisp decision and ledger write | Run once for the development-selected artifact, with at least `min_gain_fraction` absolute gain over the baseline on development and confirmation. The receipt (`natlang.acceptance-decision/1`) reports both arms' actual successes and scored counts. An exact resume is idempotent. |

Memetic callers: generations are now selected on development only. Before, every generation also had to beat the parent on confirmation, which used confirmation for selection. The selected state is measured on confirmation once and decided by `confirm`.

## Not migrated yet

- **Tuning lane split.** The TypeScript `ts-host/src/tune/collect.ts` `splitExamples` assigns tune root components without a Python process.
- **Metalearner.** Proposal acceptance and the program-family split live in `tune/learned/` (metalearner agent).

Both should call this module: the tune split via `verify_slices` over its persisted receipt, and proposals via `select`/`confirm`.
