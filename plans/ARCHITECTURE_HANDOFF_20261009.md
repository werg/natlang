# Handoff: architecture review follow-through, 2026-10-09

The owner stopped this session for usage limits. Plans: [ARCHITECTURE_IMPROVEMENT.md](ARCHITECTURE_IMPROVEMENT.md),
[REFINEMENT_TYPES.md](REFINEMENT_TYPES.md), [BATCHED_EXECUTION.md](BATCHED_EXECUTION.md).

## Owner decisions (2026-10-09)

- Clean up the code structure, using Sonnet subagents for the work.
- Rebuild the weak applications natlang-native: games, scheduling, workflow, wiki, logs, build, migration.
- Batch all pending model calls.
- Build natural-language refinement types and carry them through training, authoring and datasets.
- Replace the agent communication system: done, see below.

## On main (verified)

| Commit | What |
| --- | --- |
| de245db9 | Pyodide on Node no longer creates a stray `file:/home/...` directory; duplicated PLAN.md paragraph removed |
| a982f7ce, 338d6106 | `iteration-unbounded` error code as the spec says; `npm run check:layering` ratchet (68 cyclic directory edges frozen) |
| 42bcb73c, dbc5ec3e | `scripts/coord.py` replaces `coordination_inbox.py` (see "Communication") |
| 52aae5f1, 04417569 | Architecture plan; refinement-type and batching designs |
| cc90f3a5..184ae532 | Shared `natlang_neuralese/common/{hashing,jsonio}.py`; about 45 byte-identical copies migrated; golden digests |
| c84b93ee | 24 golden tests: LionSR, PortMuonAdamW, MuonWithAdamW, QAT ramp/STE, fused-kernel math, in-backward Lion parity |
| cce5bd6a..fc99d8ce | `common/paths.py` machine profiles (defaults unchanged); machine-path and duplicate-helper ratchets; pyproject extras `train`/`serve`/`dev` |
| bdeef99d..83ce7546 | TAT-QA/MuSiQue policy moved to `ts-host/src/benchmarks/` behind registries (ids unchanged, old paths are shims); call-store contract suite (31 tests); model HTTP mock tests (16); **call-store `evict` bug fixed** (b4c73609: it forgot bytes already freed and always deleted whole calls) |

## Communication

`scripts/coord.py` is the system. The description is in plans/MACHINE_COORDINATION.md under "Messages".

- Pop migrated and uses it (4894636f added a Python 3.10 fix).
- Claude sessions get a hook notice from `.claude/settings.json`.
- **Open request to DGX:** Pop's 2026-10-09T18:13Z request about named optimizer restore in
  `text_warmup`/`trajectories`. Today a `ValueError` silently resets the optimizer moments. The fix must preserve
  matching slots and initialize only the declared new ones. It belongs to the DGX session that owns the
  training-objective refactor. Answer with `coord.py reply <id>`.

## Unverified work in progress (branches, not merged)

Each of these was stopped mid-task. Read the branch's DECOMPOSITION.md or design notes, finish it, build and run the
tests, then rebase onto main. Do not merge any of them unreviewed. The local worktrees under `.claude/worktrees/agent-*`
hold the same state.

| Branch | State when stopped | Next step |
| --- | --- | --- |
| `wip/refinement-types` | Committed: the `refined` kind in native/types.ts (parse, format, fit with obligations). WIP: kernel/runtime check sites, intrinsics, manifest settings, spec and skill text | Finish step 2 (check sites, judge via decision scorer, cache, traces) with stub-scorer tests. Check the WIP against REFINEMENT_TYPES.md §2–3. It touches `native/agent.ts` and `runtime/kernel.ts`; confirm no clash with the other session's uncommitted `native/runtime.ts`/`runtime/hooks.ts` |
| `wip/batched-execution` | WIP only: `model/scheduler.ts`, `scoring.ts`, `server-slots.ts`, `test/scheduler.test.mjs`, edits in model/, native/agent.ts and decision.ts | Build and run the scheduler tests, plus `test/model-http-contract.test.mjs` (its requestLimit tests must still pass). Then run gates 2–3 of BATCHED_EXECUTION.md through the ledger |
| `wip/games` | Committed: natural-language rules (economy, combat, NPCs) with crisp commit checks. WIP: commit.ts edits and `test/games.test.mjs`. The agent's last note was "crisp checks pass; model-run ones fail" | Debug the scripted-model tests |
| `wip/build-migration` | Build: decomposition plus port committed. Migration: WIP | Review the build port, finish migration, run the tests |
| `wip/scheduling-workflow` | Both decompositions committed; scheduling partly implemented (WIP); workflow not started | Continue from scheduling/DECOMPOSITION.md |
| `wip/wiki-logs` | Wiki: merge/maintain/cells .nl functions WIP; logs not started | Continue; write the logs decomposition |

Per the owner's porting rule, each DECOMPOSITION.md needs owner review.

## Known issues found along the way

- `ts-host/test/inline-curriculum.test.mjs` fails on main (`TypeError: Cannot read properties of undefined (reading
  '1')`). It predates this work and has not been investigated.
- About 48 Python test failures predate this work (mainly `test_training_recipe`, `test_improvement_round`,
  `test_student_improvement_*`).
- Canonical-JSON variants without `sort_keys` remain in `scripts/self_improvement_task_inventory.py` and
  `natlang_neuralese/train/text_warmup.py`. `scripts/approve_dgx_v9_pool.py` has a hand-rolled serializer. Check
  whether any manifest hash depends on them.
- Unused exports: 14 have no reference anywhere, for example `compiler/policy.ts:80 isIterationStream`,
  `native/values.ts:344 dumpState` and `runtime/context.ts:118 contextStore`. About 270 are used only in their own
  file. Nothing has been deleted.
- `teacher/curriculum-policy.ts` and `teacher/source-review.ts` still inline dataset hold tables (treedst, qasper and
  others). They can move per dataset through `benchmarks/registry.ts`, but sealed-runtime hashes would change.
- The QAT ramp schedule is a closure inside `qat_convert.run_train`. Extract it so the golden test covers the real
  formula.

## Remaining plan items not started

From ARCHITECTURE_IMPROVEMENT.md:

- B1–B3: extract the `core/` layer, split the god files, a `ModelBackend` capability interface. Wait until the other
  session's edits to `native/runtime.ts` and `runtime/hooks.ts` land.
- B5–B7.
- C1–C3: shared training loop, one optimizer module, recipe inheritance. DGX training owns these, coordinated with
  Pop's open request.
- C5–C6.
- D1–D4.
- E1/E2/E5–E7.
- Refinement types §5 data families and the §5 live evaluation, in the teacher window.
