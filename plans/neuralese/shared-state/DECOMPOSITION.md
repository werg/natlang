# Shared-state Neuralese decomposition

Owner correction 2026-10-15. Production cutover is pending implementation and
local correctness checks; old projected-value jobs are diagnostic only.

## Natural-language policy

1. Keep every usable native trajectory, its provenance and train/held decision.
2. Identify producing and consuming spans explicitly; never infer graph edges
   from token hashes or quietly flatten a missing edge to ordinary text.
3. Train each source token once at each decoder layer. A consuming occurrence
   aliases the source's layer input KV and finite convolution inputs, with its
   own logical position. It does not invoke the source model recursively.
4. Preserve the source's private context for computing source states, while
   exposing only the declared span to a consumer. The model must learn to make
   that exported span useful without requiring the source's private cache.
5. Train the complete recipe for both models. Use native same-model state
   initially. Foreign-tokenizer text/embedding matching is explicitly best
   effort; foreign KV must never masquerade as exact native state.
6. After native training, learn per-model VAE adapters through a joint semantic
   space, then improve actual consumer behavior by distillation and SFT. An
   untrained adapter is not a usable dialect.

## Exact mechanisms and interfaces

- `model/shared_state.py`: immutable invocation/import specifications compile
  into unique physical token rows, logical KV alias slots, causal visibility,
  query positions and finite convolution predecessor gathers. Crisp because
  masks, positions, gradients and provenance must be exact.
- The same module runs existing backbone attention/FFN weights once per layer,
  using raw unrotated keys gathered into consumer positions before RoPE. This
  extends existing ports rather than introducing separate model weights.
- A low-rank residual map transforms ordinary token embeddings only for
  producing spans. No hidden-state projection or collected latent is trained.
- `forward_reference` computes each invocation in topological order using the
  same declared native state, for forward and gradient comparison. It is a
  diagnostic oracle; the production graph path is a single layer-major pass.
- Runtime storage and serving export raw KV plus finite convolution inputs,
  token text/IDs/embeddings, model/tokenizer identity and source positions.
  Exact native matching is distinct from best-effort text remapping.
- Linear/recurrent backbones other than LFM finite convolution require an
  explicit associative scan adapter; unsupported state is rejected rather than
  silently reset or described as single-pass.

## Replacement checklist

Graph compiler / masks / positions; LFM convolution; Mellum sliding attention;
forward and gradient reference parity; typed state storage; four role markers;
native runtime import/export; corpus graph conversion; shared trainer and RL
integration; full recipes; real harness samples; both cloud restarts. Each item
must have actual evidence before declaring the replacement deployed.

## Exact span and inference boundary contract

`producing=(start, stop)` and an ordinary `ImportSpan` use own-token
BOUNDARIES: imports inserted at `start` belong to the exported span, and imports
inserted at `stop` do not. `GraphPlan.boundaries` records positions before the
imports at each own-token index. Thus nested exports contain every intervening
native state alias. A `logical_producing` declaration and `ImportSpan(logical=True)`
can explicitly export a logical state range; an imported-only invocation needs
no own tokens and forwards its existing state without running the LLM. A graph
consisting entirely of forwarding performs zero token queries.

Producer begin/end marker tokens are actual trained vocabulary rows. Execute the
begin marker normally, capture payload tokens with the embedding adapter active,
then execute the end marker. The stored payload excludes those marker tokens.
Consumer begin is a normal query; splice the raw payload KV at the next logical
positions without executing alias queries; consumer end is a fresh normal query
and therefore sees the imported state. Ordinary target tokens follow that query.
Never use the source's last hidden state as if it were a consumer-close query or
train the consumer's forced close token from a source-owned logit.

Incremental `native_step` and `splice_block` use the same projections, raw-state
contract, RoPE and finite-convolution continuation as the layer-major training
path. Native imports require the exact checkpoint, expanded tokenizer, embedding
identity, architecture and state layout. Graph external state requires explicit
`external_identity`; this is also how a future semantic decoder declares its
TARGET state. It does not legitimize raw foreign-model KV.

## Evidence so far (not production qualification)

- Tiny LFM hybrid, native Mellum with per-layer YaRN/sliding windows, and Qwen:
  ordinary port parity; multi-level/fanout graph forward and adjoint parity;
  incremental capture/import versus graph parity; native channel save/restore.
- CUDA sparse block geometry matches dense attention forward and all Q/K/V
  adjoints for global and sliding attention. Production plans store linear token
  and alias metadata, not token-squared visibility/history lists.
- Actual full 350M QAD LFM, 16 layers and width 1024, FP32 CUDA: graph versus
  invocation reference max residual error 5.07e-7, embedding-adapter adjoint error
  2.28e-6, incremental native splice error 4.47e-7. This tests state/gradient
  machinery, not task reward or model improvement.
- Full Mellum, integrated serving/harness, corpus conversion and complete recipe
  execution remain required before the old diagnostic channel is retired.
