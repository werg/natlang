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
7. Keep communication learning moving while immediately exploiting explicitly
   prioritized teacher data. Start with a communication batch, then alternate
   communication and positive-priority ordinary batches until either stream
   ends. Finish the remaining stream before ordinary priority-zero data. Within
   each stream retain declared priority, source order and visit order. A changed
   schedule never changes row quotas, earned exposure or the fixed communication
   foundation; completion still requires every declared visit.

The shared `pending_exposure_batches` function implements step 7 as exact batch
packing and credit accounting. The source manifest's existing `training_priority`
is the policy input; this adds no model-facing text or independent scheduler.

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

## Native arithmetic and gradient contract

The native channel is `neuralese.native-channel/2`. Its low-rank embedding
adapter is FP32; shared raw KV and finite-convolution inputs meet at FP32 adjoint
junctions, then the executed attention/convolution receives the declared model
activation dtype. This sums fanout gradients before the single cast back to the
source layer. The native forward does not project states into a communication
latent. Ordinary token-local linear operations reuse the existing fixed-tile
`maple.fused_moe.invariant_linear`, including its input adjoint. Custom adapter or
quantized modules keep their own forward implementations.

Each invocation's KV slots begin on a 128-token block boundary. The same native
Flex training kernel executes graph and cached attention, so cache imports do not
change BF16 softmax tile boundaries. Padding creates no token queries and is
always masked. The independently compiled kernel contracts distinguish actual
architecture/dtype/cache layouts while leaving sequence lengths dynamic. A
full-graph compiler contract forbids silent unfused dense attention: Dynamo's
previous shared eight-specialization cache otherwise changed both arithmetic and
memory behavior after another model or shape ran in the process.

FP32 attention uses IEEE arithmetic and the kernel checks use an FP64 dense
oracle with identical upstream gradients. Squared losses evaluated separately
on two rounded outputs are not used as a test of the attention input adjoint.
Destination startup additionally declares IEEE FP32 matrix/convolution precision;
H100's NGC defaults otherwise enable TF32 in those FP32 qualification paths.
Tiny LFM and Mellum BF16 graph/cached forward checks are bitwise equal and their
actual consumer CE/embedding-adapter gradients pass the same tight checks in one
process after mixed attention contracts. Full-size destination qualification is
still recorded separately from unit correctness and task quality.

## Attention backward correction, 2026-10-15

The kernel contract is `shared-state-fixed-native/2`. Query origins, as well as
KV origins, align each invocation to 128 rows. Only attention Q/output workspace
is padded: residuals, embeddings and FFNs still execute exactly
`GraphPlan.token_count` rows. `GraphPlan.attention_workspace_rows` separately
reports this allocation; padded workspace is never data exposure. Dummy Q rows
attend a finite key within their own invocation and receive zero adjoints after
the physical-row gather.

A second, independent defect was the compiler-generated Flash backward delta
`sum_D(out * grad_out)`: its reduction layout changes with query/head counts.
That can change source-key adjoints while outputs, LSE and incoming adjoints
are bitwise equal. Raw auxiliary LSE routing initially fixed full Mellum but
failed a reduced-head counterexample; that routing alone is not the repair.

`common/native_attention_delta.py` supplies one fixed FP32 per-row Triton
reduction. A native-only, fail-closed compiler lowering substitutes that delta;
attention itself keeps its existing sparse, tensor-core Flex forward/backward.
There is no graph replay or dense mask fallback. Lowering-local dispatch avoids
temporary mutation of compiler arithmetic while another model compiles.
The checkpoint records query alignment, fixed-delta version and raw-log2-LSE
auxiliary contract, and rejects the earlier diagnostic kernel contract.

Final fixed-delta receipts (DGX, 2026-10-15): actual full 28-layer BF16 Mellum
has bitwise forward/logits/consumer CE, source-adapter gradients, incremental
splice outputs, and all layer backward adjoints against cached invocation
execution. Report: `dgxops/native-mellum-fixed-delta-proof-20261015.json`
(141.78 s including model load; 28.55 GB peak CUDA allocation). Actual full QAD
LFM through the published qualification CLI passes unchanged 1e-4 forward and
1e-5 gradient bounds: forward 2.086e-6, source-adapter gradient 1.732e-6,
incremental 7.153e-7; optimizer update and exact channel restore pass.
`dgxops/native-lfm-fixed-delta-proof-20261015.json` records that receipt.
These are numerical/runtime proofs, not task-performance or training-lift claims.

The focused shared-state suite passes 41 checks. The seeded 8/32-query-head
regression checks every Q/K/V adjoint bitwise; disabling the fixed delta causes
the 32-head counterexample to fail. Reduced actual Mellum head fixtures also
pass on both auxiliary-output routes. Delta-only timing at 32 heads, D=128:
0.0185 ms (128 queries), 0.0573 ms (1024 queries), versus the compiler reduction's
0.0166/0.0255 ms. The additional per-layer preprocessing cost preserves sparse
attention scaling and does not add hidden-state or FFN padding.

The lowering explicitly checks its expected PyTorch delta expression and fails
closed on incompatible compiler changes; qualification on each destination
PyTorch/CUDA build remains required. Standing full-suite checks and independent
integration review remain separate, as do the model-facing marker debt and
actual full-recipe cloud restart.
