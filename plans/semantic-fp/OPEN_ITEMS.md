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

All of these are owned by the language-gaps agent (2026-11-23 15:00):
- `natlang improve DIR` only checks and adopts a given proposal. It should propose changes from the program's own runs (WS8 N4).
- The `escalate` NL policy doesn't read escalation history from the call store; the caller passes it (WS2 R4).
- The `.d.nl.ts` `import type` emit for `uses:` modules is not built (WS1 L6).
- An inline `nl.with<T>` built inside a callable-folder TypeScript factory shows its parameter as `unknown`.
- `natlang check --idiom` doesn't call `applications/idiom-reviewer`.

## Applications

These are owned by the language-gaps agent after the gaps above:
- codemod-studio re-implements the `applications/migration` edit/check/repair loop. It should call it.
- sem-grep-sql doesn't build on `applications/nldb`.
- ci-doctor's bisect has no natural-language stop, and its failures are not clustered semantically.
- diligence has no Neuralese views or tuning.

## Held and data

- **held-v4:** no `branch_and_select` task; B1's 104 rows keep `invented:` groups; the 100 rubric workflows and 2 reserved apps are outside the set. Owner: next held-v4 extension; due before the first held-v4 cadence measurement after round 3.
- **Flagship held splits** are 2–5 fixtures per app, too few for p < 0.05. Owner: task-author tool batches, next semantic-FP batch.
- **`compatible.nl`** judges additive edits incompatible. This is a training target through the sfp3 rows, which are in the pool; watch its family in the health report.

## Operations

- **`vibe-native-v9-reader-pin*` units failed at 08:15 on 2026-11-23:** `.venv-decider-systempython conflicts with the intended shared root link` in `archive-code-00ab72d28dbc`. The Vibestudio session later reached v9 through other units (`vibe-student-fill-keeper-v9`, `vsingest-semantic-poll-v9`), so this is superseded; the owner is the Vibestudio session.
- **The Colab keeper's real replacement and upgrade path is untested live.** It is watched by the ops watchdog (`colab` checks).

## Deferred on purpose

- **A preempted cycle collection gets a fresh `--hours` window:** affects only the legacy cycle path, not async rounds.
