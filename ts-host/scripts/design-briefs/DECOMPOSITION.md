# Design-brief corpus preparation

## Responsibility and boundaries

The 100 supplied application briefs are implemented under
`codebases/design_briefs/`. Their typed natural-language functions own domain
interpretation, evidence selection, policy precedence, stage construction,
counterinterpretation, reconciliation, change impact, and review planning.
Every application preserves its five domain algorithms and the five operational
details from its source brief. The host never substitutes an answer table for
those judgments.

This directory supplies exact build and verification mechanisms, not another
semantic implementation:

| Component | Kind | Responsibility |
| --- | --- | --- |
| `lib.mjs` | Exact mechanism | Read confined source trees; bind their hashes; load native definitions; validate typed fixtures; assign the declared whole-application split. |
| `build.mjs` | CLI and exact serialization | Freeze the authored source and independent fixtures as `natlang.program/2`; produce separate exact-gate and review-required workflow packets plus a content manifest. |
| `check.mjs` | Compiler adapter | Run the repository's actual `checkProject` on every application, with generated declarations kept out of source. |
| `prepare.mjs` | CLI and exact plumbing | Verify the frozen packet manifest and invoke the existing RL queue builder in student-root mode. Write teacher launch arguments without launching a model. |
| `check.test.mjs` | Engineering tests | Exercise path confinement, split closure, oracle/fixture contracts, deterministic packaging, and tamper detection. Scripted checks do not establish model quality. |

## Data flow

The build reads native source, brief metadata, and independently stored case
inputs/oracles. Only `.nl` and non-generated `.ts` files enter the model-visible
project. Hidden metadata and expected answers remain outside that project.
The existing native loader resolves every function's callable folder and types.
The runtime's own type coercion checks fixture inputs and exact expected values.
`operational-contracts.json` retains section 9's five original use cases for each
brief. The build requires a one-to-one mapping from those use cases to the five
workflow fixtures, and separate assertions for the view, proposed durable state,
and proposed effects. Stage descriptions cannot substitute for these use cases,
and observed evidence cannot simply repeat their prescribed actions.

Related task variants never cross a split. Within each five-application domain,
the first three applications are TRAIN, the fourth is development, and the fifth
is held evaluation. This conservative split is declared before rendering and
holds an entire application family out; a paraphrase or repaired variant retains
its original application group.

Exact finite qualification cases are preparation inputs for RL. Full workflows
use independent semantic rubrics; they remain review-required. The build creates
no reference assistant responses, sampled actions, native-state receipts,
teacher trajectories, trained checkpoints, or claims of learner exposure.

Teacher collection uses `teacher-collector.mjs` with the configured self-hosted
teacher. RL uses `natlang_neuralese.rl.task_rollouts prepare --root-mode student`;
it must retain the policy's actual complete execution graph when collected.
No minimum communication-edge count is an eligibility requirement. Any genuine
producer/consumer state is retained and differentiated through the existing
whole-graph path. Source preparation does not resume paused services.

## Validation and remaining measurement

Static inventory, compiler, fixture typing, deterministic hashes, and adversarial
oracle checks precede publication. The corpus is a source candidate until actual
self-hosted teacher measurement and source-quality review complete. Those steps
are tracked separately from admission and training exposure. No existing
protected evaluation data is used to author these new worlds.
