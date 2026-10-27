# Joint learned optimizer: decomposition

Implements [plans/neuralese/JOINT_LEARNED_OPTIMIZER.md](../../../../../plans/neuralese/JOINT_LEARNED_OPTIMIZER.md).
Each responsibility of the plan is listed with its form (Natlang function, one instruction, implicit, or crisp
helper) and the reason. Natural-language policy lives in the Natlang program
[`ts-host/src/tune/optimizer/`](../../../../../ts-host/src/tune/optimizer/); this package holds the crisp mechanism
it calls through services: exact tensor math, data custody, execution, storage and verification.

| Part | Form | Where | Why |
| --- | --- | --- | --- |
| Interpret evidence: focus, key evidence, digests | NL function `interpretEvidence` (350M) | `optimizer/interpretEvidence.nl` | Semantic judgement over failures; trained with hindsight labels and whole-graph gradients |
| Exact evidence facts (pass/fail, first failing binding, error kind, verified vs attempted action, counts) | crisp helper | `evidence.py` | The runtime knows these exactly; no model guesses them |
| Compact evidence view, permitted-visibility filter | crisp helper | `evidence.py` | Causal visibility is an exact data rule |
| `keyEvidence` default rule | crisp baseline behind pluggable | `evidence.py` `key_evidence_rule` | Failed items of focused bindings, then their demonstrations; NL variant learns |
| Choose revision operations per focused binding | NL function `chooseRevision` | `optimizer/chooseRevision.nl` | Policy over contracts and dependency structure |
| Write instruction/skill states | native writer (contextual re-encoding) | `writer.py` | Differentiable capture through `native_step` at the target's slots |
| Weight arrays | NL computation, token head + coefficient head | `writer.py`, `numbers.py`, `head.py` | Visible integer grid tokens; bounded differentiable refinement |
| Choose operator (learned proposal, retrieval, refinement, stop) | NL policy behind pluggable | `optimizer/chooseOperator.nl`, `policies.py` | Memetic strategy learns; crisp baseline uses the same interface |
| Stop | NL predicate with structural termination | `optimizer/adapt.nl` (`iterateOn` measure) | Benefit/cost decision; host measures cost exactly |
| Shared basis, component bank, keys, snapshot IDs | crisp | `bank.py` | Exact math and immutable identity |
| Gradient subspaces and probes | crisp | `subspace.py` | `U^T G V` from streamed activations and output gradients |
| Bank views and dropout masks | crisp mechanism, pluggable selection | `views.py` | Exact positional mapping; selection policy pluggable |
| Array validation and composition into adapters | crisp | `views.py`, `bank.py` | Cross-value constraints a `number[]` type cannot express |
| Bundle manifest, proposal receipt, release | crisp | `bundle.py` | Atomic, content-addressed, exactly composed |
| Objectives (distill, state, mixture CE, expectation, decoded weight, head regression, structure) | crisp | `objectives.py` | Exact losses with reported gradient shares |
| In-bank fits and focus hindsight labels | crisp | `fit.py` | Gradient fits through the exact composition |
| Program context from the tuner's plan and build (rows, soft init, scopes, views, incoming bundle) | crisp | `program.py` | One representation shared with `tune.build`: the same plan, turns, soft states and scope tree |
| Target records (one schema, three tiers) and tuner export | crisp | `assemble.py`, `targets.py` | Custody, lineage and exposure; free tunes on the bank basis are exact targets |
| Corpus: tuner fits on the bank basis per program, then labels | crisp | `__main__.corpus` | Operations over existing tune plans; resumable |
| Program supply (typed decision programs with exact oracles) | crisp generator | `tune/supply.py` | Exact oracles and grouped worlds; function texts are model-facing and measured in tuning |
| Meta-RL estimators (paired reward, shaping, filtered BC, joint ratio) | crisp | `meta_rl.py` | Exact policy math, reused from `rl/policy_math.py` |
| Ceiling ladder, evaluation power, planted controls | crisp | `ladder.py`, `controls.py` | Exact measurements |
| Outer training step and resumable state | crisp | `trainer.py` | One snapshot, cotangent replay, FP32/AdamW, exposure |
| Select training work, adapt loss mixture | NL policy with crisp baseline | `optimizer/scheduleTraining.nl`, `policies.py` | Operations decision; quotas and receipts stay crisp |
| Entrypoint | crisp CLI | `__main__.py`, `natlang tune --learned` | Resume the whole operation from a declared recipe |

## Gradient path

The writer runs under its own adapter stack (`run_native` with the writer entries), the executor under each scope's
composed adapter. One forward cannot hold both because the executor's adapter depends on the writer's outputs. The
outer step therefore uses exact cotangent replay, the plan's sanctioned split-process form:

1. Writer forward (no grad) produces states, slot hidden states and branch logits; detach them as leaves.
2. Executor objectives run with those leaves as live state overrides and composed-adapter leaves (cores are a
   differentiable function of the head outputs and the trainable bank); backward yields their cotangents and the
   bank/basis gradients.
3. Writer forward again under its adapter with gradients; `backward(outputs, cotangents)` plus the writer-local token
   losses. `tests/neuralese/test_learned_optimizer_writer.py` checks equivalence against a single-actor graph.

Parsed integers never carry a gradient; the token-expectation estimator and the coefficient head are the declared
continuous paths.
