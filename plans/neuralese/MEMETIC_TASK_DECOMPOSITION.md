# Memetic optimiser on natlang task success: decomposition

Status: retired historical decomposition (2026-10-23). Its flat soft-skill and
weight-adapter entrypoints were removed. The native repeated-invocation program
path and current command are in [MEMETIC_NATIVE_PROGRAM.md](MEMETIC_NATIVE_PROGRAM.md).

## Goal

The memetic optimiser so far searched decision readouts (`memetic-decision.mjs`). This version changes the
fitness to **natlang task success**. An individual is a program state:

- crisp guidance text,
- soft values,
- a weight adapter.

It is scored by running train-split task rows end to end through the collector and classifying each rollout with
the held eval's own verifier (`eval.task_packet.classify_case`, via `rl.task_rollouts.score`). The held 48 are never
used. `authored_root_row` refuses any row that is not `split == "train"`.

## Natlang-native design

The optimiser is a program with two kinds of part:

- **Decisions** are natural-language functions:
  - what to try next;
  - how to rewrite the guidance from failures.
- **Mechanism** is crisp:
  - running rollouts;
  - verifying outcomes;
  - computing a gradient step;
  - keeping records.

| Unit | Becomes | Why |
| --- | --- | --- |
| Run an individual on the rows | crisp helper (`evaluate`) | Outside world (collector, server), exact. Reuses `task_rollouts.build_queue` and `run_bonsai_queue.py` with per-entry `system_file` and a per-runner `--chat-request-config` for adapters. Common random numbers: every individual runs the same rows with the same seeds. |
| Score a rollout | crisp verifier (`classify_case`, `reward_of`) | An exact verifier of what the stage produced. It is shared with the held eval and S7, so one implementation per concept. |
| Fitness of an individual | crisp (`fitness`) | Arithmetic over verified rewards: successes / scored, with model-request budget stops counted as failures and infrastructure unscored. |
| Failure digest | crisp helper (`failure_digest`) | Exact extraction of each failed rollout's task text, outcome class and rejection or error detail. It is data, not a decision. |
| Rewrite guidance (`propose`) | **NL function** `proposeGuidance` | A policy. The author model reads the digest and the current guidance and returns new guidance (numbered steps for a small executor). |
| Choose parent and operator | **NL function** `chooseNext`, behind a pluggable (`--policy crisp\|nl`) | A policy. The crisp default is a tournament between the two best plus a UCB bandit over operators. The NL version receives the population table and the operator history and must answer with one parent ID and one operator from the allowed list; an invalid answer falls back to crisp and is recorded. |
| Survival | crisp (`select`) | Keep the top `population` individuals by fitness, ties broken by fewer budget stops. This is a total order on verified numbers, not a judgement. An NL survival policy is a later unit. |
| Gradient operator (`refine-adapter`) | crisp mechanism (`grpo.py round`) | Group-relative advantages from the individual's own verified rollouts, replayed as `logLikelihood` terms; the result is a new adapter block. It needs groups with spread (samples ≥ 2, temperature > 0). |
| Soft values (soft system-prompt pieces, soft skills) | **not yet** | The collector has no channel for soft blocks in the system prompt, only text and adapters. Next unit: a `x_natlang_system_prompts` chat-request control that the server resolves as `withSystemPrompts` does, then `embed` and `refine-soft` operators as in `memetic-decision.mjs`. |
| Records | crisp | One `natlang.improvement-step/1`-shaped line per operator application in `steps.jsonl`, and `population.json` after every generation. Rollouts are kept as `natlang.rollout/1` (`rollouts.jsonl` per evaluation). |

## Channels into a run

- **Guidance text.** The collector's `--system-file` is the default system prompt followed by
  `<natlang_program_guidance>…</natlang_program_guidance>`, the runtime's own program-guidance wrapper (`adaptation/prompts.ts`).
  It is set per queue entry with `system_file` (run_bonsai_queue.py).
  Under `program-header` rendering, generic pieces are dropped and unmatched text is kept verbatim, so the guidance
  reaches a direct-SFT student too. This was checked in the smoke from the recorded contexts.
- **Adapter.** `x_natlang_adapters` in the runner's `--chat-request-config`, as in S7.

## Loop (numbered, as the driver runs it)

1. Prepare the packet from `--rows` train rows at `--offset`.
2. Evaluate the seed (empty guidance, no adapter).
3. For each generation:
   1. Ask `chooseNext` for a parent and an operator.
   2. Apply the operator:
      - `propose`: `proposeGuidance` from the parent's failure digest, keeping its adapter.
      - `refine-adapter`: one GRPO round on the parent's rollouts, keeping its guidance.
   3. Evaluate the child on the same rows and seeds.
   4. Credit the operator with the fitness gain.
   5. Run survival.
   6. Write the records.
4. Report the population, the best individual and the per-row success table.

## Not in this version

- Soft values (above).
- Held-out confirmation of the winner on a disjoint train slice (next).
- NL survival.
- Merge of relatives.
- Neuralese consumer values in the tasks themselves (the e2e stage).
