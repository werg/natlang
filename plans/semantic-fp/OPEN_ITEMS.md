# Open items register

This register lists every known open item, each with an owner or a deliberate deferral. It is kept by the coordinator and reviewed at every 4-hourly review.

An item leaves the register when its fix lands on main. A finding reported by an agent enters here the same day, unless that agent is fixing it itself.

## In flight

| Item | Owner |
|---|---|
| B10 four-engine smoke, fork pin move to b12088 (reasoned decisions on llama-server and wllama), ENGINE_PARITY rows | B10 agent |
| Decision option scoring under the standard tokenization only. The model may generate another split (`"`,`spam`,`"` vs `"sp`,`am`). Sum both tokenizations per option on all four engines and the learner (8b34839ab) | B10 agent, after the fork release |
| Tuple `prefixItems` constrained decoding on all four engines | tuple-parity agent |
| M2 migrator: one shared `stepKind`/`instructionsOf` in std, `migrate.nl` replaced by a TypeScript driver, P1/P2 measurements, pairs dataset | M2 agent |
| Benchmark arm D on sft-v9 (specialize, tune, escalate); then regenerate `examples/triage/bench/results/2026-11-23.json` with D | armed driver `semantic-fp-bench/triage-20261123b/driver.sh` (waits for the policy server's 18.9 GB ledger claim); the coordinator commits its `results-ABCD.json` |
| Memetic layout fix (c64d250b2) on a real actor | queued unit `memverify-dgx3`, then the round 1 health report |
| held-v4 sft-v6 reference measurement | Pop unit `popheld-hv4-sft-v6`; the coordinator appends the result to DECISIONS |
| Remove the legacy `expert_iteration.py cycle` mode once aei1's round 1 has a held verdict (AGENTS.md: async-drive replaces cycle) | coordinator, after the r0001 verdict |
| aei1 round 1 build, SFT and held verdict | aei1 driver; the coordinator reads its health report |

## Reasoned decisions in training

Found by the GRPO-reasoning agent (DECISIONS 2026-11-23 "GRPO trains the action reasoning"); the coordinator assigns both:
- **The scorers re-tokenize the reasoning.** Every engine scores the options after the reasoning *text*, re-encoded, not
  after the sampled ids. In aei1 r0000 (16 action readouts, groups g000689/696/701/720) the scored prompt leaves the
  sampled ids mid-reasoning in 4 of 16 (after 847 of 3476, 531 of 1051, 535 of 2085 and 953 of 1687 tokens; the decoded
  text is the same). The choice is then scored, and trained, after a tokenization the model never produced. Fix: the
  score item carries the sampled reasoning ids, and every engine (Python native, SGLang, llama.cpp fork, wllama) uses
  them. GRPO's lowering counts it each step (`decision_reasoning_ids_rendered` in `train.jsonl` `lowering`).
- **Tune RL with reasoned readouts.** `ts-host/src/tune/rollouts.ts` pushes every reasoning generation as an ordinary
  turn (the `turn` wrapper), and `tune/rl.py` passes them to `native_replay.prepare_objective` unfiltered; only the SFT
  path (`tune/rows.lower_rows`) skips them. The graph has one node per scored entry, not per reasoning generation, so
  the turn names cannot match it. Expected effect: such episodes are refused, or all k reasonings and their unused free
  replies train. Not run. Fix: tune rollouts keep only the action sample's reasoning, as `rl.task_rollouts` does now.

## Language and runtime gaps

Closed 2026-11-23 by the language-gaps agent (DECISIONS "Semantic-FP language gaps and applications closed"):
`natlang improve DIR` proposes from the program's runs; `escalate`'s nl policy reads its history from the call store;
`.d.nl.ts` `import type` for `uses:`; `nl.with<T>` in callable-folder factories typed from `import type`; `natlang check
--idiom`. Found while closing them, not changed:
- **`natlang improve DIR`:** memetic guidance changes are not a transformation yet (WS8 N4 names them); the improver
  revises `.nl` instructions only. Owner: language-gaps agent's next session, or N5.
- **`Untrusted<T>` in the migration loop:** codemod file texts stay fenced (the fence is keyed by text), but a site
  excerpt or an edited revision of an untrusted file is new text and is not marked. Owner: the untrusted-contract
  follow-up.
- **Type-only imports in callable-folder TypeScript** that resolve to nothing still make the type `any` silently (the
  module's TypeScript diagnostics are not reported). Owner: language-gaps agent.

## Applications

codemod-studio (now `applications/migration/codemod`, on the migration loop), sem-grep-sql (now `applications/nldb/grep`,
on nldb's SQLite engine) and ci-doctor (natural-language bisection stop, root-cause clustering) are closed. Open:
- diligence has no Neuralese views or tuning.
- `ts-host/test/repository-migration.test.mjs` keeps its own copy of the scripted migration stages
  (`test/support/migration-stages.mjs` is the shared one); fold it in when that test next changes.

## Held and data

- **held-v4:** no `branch_and_select` task; B1's 104 rows keep `invented:` groups; the 100 rubric workflows and 2 reserved apps are outside the set. Owner: next held-v4 extension; due before the first held-v4 cadence measurement after round 3.
- **Flagship held splits** are 2–5 fixtures per app, too few for p < 0.05. Owner: task-author tool batches, next semantic-FP batch.
- **`compatible.nl`** judges additive edits incompatible. This is a training target through the sfp3 rows, which are in the pool; watch its family in the health report.

## Operations

- **Round builds read archived teacher pools file by file from the external HDD:** `/mnt/external/dgx-ops-archive-20261118/teacher-gen-20261115/pool-*`. r0001's build spent about 2 h in I/O wait there. Index the pools once onto NVMe, or pack them as one sequential file per pool, before r0002's prepare. Owner: coordinator, after the r0001 build.
- **Both disks are below their floors (2026-11-23 16:18; `/` 38G, external 48G).** Run a cold-data pass on `/mnt/external`: write the path list first, never touch other sessions' archives, and run it after the r0001 build so it doesn't compete for the HDD. Owner: coordinator.
- **`vibe-native-v9-reader-pin*` units failed at 08:15 on 2026-11-23:** `.venv-decider-systempython conflicts with the intended shared root link` in `archive-code-00ab72d28dbc`. The Vibestudio session later reached v9 through other units (`vibe-student-fill-keeper-v9`, `vsingest-semantic-poll-v9`), so this is superseded; the owner is the Vibestudio session.
- **The Colab keeper's real replacement and upgrade path is untested live.** It is watched by the ops watchdog (`colab` checks).

## Deferred on purpose

- **A preempted cycle collection gets a fresh `--hours` window:** affects only the legacy cycle path, not async rounds.
