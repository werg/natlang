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
  values are referenced by block ID and by the node that produced them.

| Kind | Records |
| --- | --- |
| `invocation` | Definition ID and revision, context ID, signature, argument references, capture bindings (snapshot or live), outcome. |
| `context_bind` | Function, old and new context IDs, the interface check result. |
| `model_turn` | Rendered input as text and block references, sampled tokens, sampling settings, seed. |
| `block_write` | The model turn it belongs to, the context position, every stop decision with its probability, final length, `truncated`, resulting block ID. |
| `block_read` | Block ID, the model turn, the payload positions. |
| `readout` | `read` call: input block, produced value, validation result. |
| `combinator` | Operator, operator body revision, inputs, output. |
| `rewrite` | Rule, site, enabling comparison, the nodes it replaced. |
| `effect` | Service, method, arguments, recorded result. |
| `iteration_step` | Iteration ID, step index, state references, predicate result, review verdict. |
| `grad` | Loss node, argument references, order, resulting gradient reference. |
| `literal` | A model-written literal in eval code: the block, its contextual type, the reference expression it was replaced with. |

Block payloads are not stored in the record; they are referenced by ID and kept in
the tensor store or `.nz` files for as long as a record that references them is
retained.

## Discrete and continuous

Sampled text tokens, tool choices, stop decisions, readout choices and host
branches are **discrete**. Payload vectors, projections, soft bodies and soft
values are **continuous**. Gradients flow only through continuous parts. Discrete
choices are trained through `logLikelihood` (supervised imitation or reward-weighted
policy gradient), never by differentiating the choice.

## Replay

A trainer, or a server running `grad`, replays a record:

1. Restore the definitions, context revisions, operator bodies and model revision
   used at each node.
2. Re-run the model on each `model_turn` with the recorded discrete choices forced:
   the same tokens, the same stop positions, the same tool calls.
3. Supply recorded `effect` results instead of repeating effects.
4. Recompute every continuous quantity with gradients enabled, from the soft
   arguments of `grad` through writes, reads and readouts to the loss.
5. Accumulate gradients at each producer from all of its consumers.

Nested `grad` is first-order by default: inner gradients are treated as constants.
With `{ order: 2 }` the inner replay is itself differentiated. `stopGradient(v)`
marks `v` as a constant.

A record whose discrete choices would differ under the current values is still a
valid replay for the recorded trajectory's likelihood, but its observations are not
evidence for a different choice. Changed choices need fresh rollouts.

## Retention

Records are training data. They follow the lineage, licence and split rules of the
training pipeline ([NEURALESE_DATA.md](NEURALESE_DATA.md)), and records built under
different runtime versions are not mixed.
