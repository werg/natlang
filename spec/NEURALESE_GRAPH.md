# Execution graph and replay

Version `natlang.exec-graph/1`, 2026-10-03. Normative for runtimes, servers and
trainers. Part of the Neuralese section of [SPEC.md](SPEC.md). Schema:
[neuralese-graph.schema.json](neuralese-graph.schema.json).

The native trace (`ts-host/src/native/trace.ts`, `reduction-trace/1`) is a sequence
of events. The execution graph extends it with typed nodes and edges, so that every
soft value can be traced to its producer and every consumer, and a recorded
execution can be replayed for gradients.

## Record

A graph record is the trace's event sequence plus:

- **Manifest additions:** graph version, model identity and revision, dialect
  version, enabled rewrite rules, root context ID, seeds.
- **Nodes:** events with a `node` ID and a `kind` from the table below.
- **Edges:** each node lists its inputs as `{ node, port }` references; soft
  values are referenced by native state ID and by the node occurrence that
  produced them. A repeated content ID is bound to its actual producer
  occurrence, never matched by equal text or equal ID alone.

| Kind | Records |
| --- | --- |
| `invocation` | Definition ID and revision, context ID, signature, argument references, capture bindings (snapshot or live), outcome. |
| `context_bind` | Function, old and new context IDs, the interface check result. |
| `model_turn` | The turn, sampling temperature and seed (null for a decision readout), completion token count, the tool calls made, and its inputs: the invocation and every native state the turn consumed. |
| `block_write` | The model turn it belongs to, the resulting native state ID, `format: "native-state"`, the writer's native identity (architecture, checkpoint, tokenizer, embedding), token length and width, the tensor inventory, the SHA-256 of the span's text, the source kind and the declared result or argument type; for a forced text write, the server's source capture. `stops` is empty and `truncated` is false for a closed span: a producer span ends at the model's end marker. |
| `block_read` | Native state ID, the model turn, the state's writer identity (architecture, checkpoint, tokenizer, embedding), `stop_gradient` for a constant use, and the import position. Normally `positions` is `[message, part]` in the request; when `position_source` is `captured_prompt_segment`, it is `[0, segment_index]` in the server's exact captured prompt. |
| `readout` | `read` call: input state, produced value, validation result. |
| `combinator` | Operator, operator body revision, inputs, output. |
| `rewrite` | Rule, site, enabling comparison, the nodes it replaced. |
| `effect` | Service, method, arguments, recorded result. |
| `iteration_step` | Iteration ID, step index, state references, predicate result, review verdict. |
| `literal` | A model-written state in eval code: the state, its contextual type, the reference expression it was replaced with. |

A gradient session is not a graph node: the runtime records each one as a replay record
(`natlang.replay-record/1`, `ts-host/src/neuralese/replay-records.ts`) holding its arguments, terms, order,
native turns and graph nodes, and the resulting gradients.

Node IDs are unique within a task: an invocation is `call:<call id>`, any other node `<call id>#<n>`. An input
whose producer is outside the record (a `.nz` file, another task) names the node `external` and the state.

Native states are not stored in the record; they are referenced by ID and kept in
the native archive or `.nz` files for as long as a record that references them is
retained.

## Discrete and continuous

Sampled text tokens (including the tokens of producer spans and their end
markers), tool choices, readout choices and host branches are **discrete**.
Model parameters, native adapter coefficients, and the KV and hybrid state that
a producer computes are **continuous**. A native state is a deterministic
function of its producer's context, its sampled span tokens and the current
weights, so gradients flow from each consumer through the spliced state back
into the producer's computation. An input with no recorded producer keeps its
exact archived state, which can itself be a gradient root. Discrete choices are
trained through `logLikelihood` or `policyGradient` (supervised imitation or
reward-weighted policy gradient), never by differentiating the choice.

## Replay

A trainer, or a server running `grad`, replays a record:

1. Restore the definitions, context revisions, operator bodies, model checkpoint
   and adapter bindings used at each node, and every archived native state the
   record names. A recorded turn whose actor differs from its captured binding
   is rejected.
2. Re-run the model on each `model_turn` with the recorded discrete choices forced:
   the same tokens, the same producer spans, the same tool calls. The recorded
   prompt must equal the rendering of the recorded messages, including each
   consume-marker import.
3. Supply recorded `effect` results instead of repeating effects.
4. Recompute every producer and consumer in one teacher-forced native graph
   pass with gradients enabled, splicing each produced state into its
   consumers at the recorded positions, from the gradient roots (archived
   states, adapter coefficients or named model parameters) to the loss.
5. Accumulate gradients at each producer from all of its consumers. Each root's
   gradient is returned as a named tensor artifact bound to that root and the
   exact checkpoint identity.

Nested `grad` is first-order by default: inner gradients are treated as constants.
With `{ order: 2 }` the inner replay is itself differentiated: the runtime records
the inner gradient sessions and optimiser steps the loss ran (in order, with the
artifacts they produced) and sends them with the outer objective as one
`natlang.native-derivative-program/1`; the server recomputes each as a function
of the outer roots (inner gradients with a graph of their own, optimiser steps
applied to those tensors) and lets each derived artifact stand for its recorded
value in everything after it. Losses and observations stay the recorded ones.
Derivatives beyond order 2 are rejected. `stopGradient(v)` marks `v` as a
constant.

A record whose discrete choices would differ under the current values is still a
valid replay for the recorded trajectory's likelihood, but its observations are not
evidence for a different choice. Changed choices need fresh rollouts. A state
imported from outside the record keeps its recorded tensors; changing the
weights or adapters that produced it requires a fresh rollout unless an explicit
`same-base-adapter-state` mode is declared.

## Retention

Records are training data. They follow the lineage, licence and split rules of the
training pipeline ([NEURALESE_DATA.md](NEURALESE_DATA.md)), and records built under
different runtime versions are not mixed.

A finite native decision may be a `model_turn` with `readout: "decision"`. It
records an actual deterministic conditional readout, with null temperature/seed
and no sampled tool calls. Its native receipt contains the rendered prompt and
each option's token continuation, and its block-read edges retain actual source
occurrences. A decision receipt is not a sampled-policy density.
