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

On an operational stop, finish the current optimizer update, retain its exact
exposure credit and save the full resumable state. Do not start a held/task
evaluation or runtime qualification after that request, including a request
received during a held check or checkpoint write. Completed exposure remains
recorded, while qualification waits for a resumed run. Stop checks and atomic
checkpoint durability are crisp mechanisms for this owner policy.

## Reuse identical ordinary causal prefixes

Read-only sampling of 1,000 admitted Qwen V5 graphs found that 72.4% of physical
token queries repeat an identical prefix inside the existing 32,768-token
batches; the corresponding evidence-header sample showed 6.7%. These are
potential backbone savings, not measured end-to-end speedups.

1. Retain each original example, target weight, normalization and exposure.
2. For an ordinary batch with one unadapted invocation per example and no
   imported or external state, form a compact trie of exact token prefixes.
3. Execute each trie segment once. Children import the complete raw causal
   prefix at its original positions through the existing KV/convolution alias
   mechanism. Cache exports are explicitly distinct from Neuralese-producing
   spans: they never activate the learned embedding adapter or earn
   communication credit.
4. Gather the original per-example prediction queries from their shared physical
   rows. Retain all targets, including the first target after every branch,
   duplicate examples, and examples ending inside another example's prefix.
5. Keep authenticated communication graphs and policy objectives on their
   existing graph layout. Exact full-context identity is required for sharing;
   matching text in a different state is insufficient.
6. Compare logits, normalized losses, gradients and actual optimizer updates
   against disjoint packing on both backbones before production adoption. Keep
   an explicit packing switch for diagnosis and continuation.

Trie construction, raw cache-export validation, query gathers and numerical
comparison are crisp mechanisms for this policy. No corpus, prompt, tokenizer,
source authentication, quota or inference transport changes are implied.

Local qualification: 13 focused cases passed for both ports, including logits,
all gradients, two actual MuonSR updates with matched per-matrix rounding draws,
duplicate/interior-terminal targets and old RL graph identities. The existing
96 shared-state/trainer/optimizer cases also passed. A separate actual trainer
test passed a cached update and an uncached continuation with exact earned row
counts and retained optimizer history. The packing switch defaults off pending
full-size CUDA qualification and a useful throughput measurement.

The initial optimizer comparison found different BF16 momentum bits despite
close raw FP32 gradients. Every stored coordinate is now checked exactly against
its own previous BF16 momentum promoted to FP32, the actual gradient and the
final BF16 cast. This accounts for cast-midpoint differences and their recurrence;
raw-gradient, actual-weight and other optimizer-state tolerances were not widened.
This proves matched-draw update correctness, not bit-identical production random
draw scheduling or identical rounded momentum across different graph layouts.

## Exact mechanisms and interfaces

The existing selected expert-backend policy owns both precision and dispatch:
an FP8 training scope selects grouped expert execution, and replay restores that
same selection after the outer scope exits. Native evaluation disables FP8.
Only successful FP8 projection calls count as executed work; configured module
counts and entering a precision scope do not attest execution. These scoped
settings, restoration and counters are crisp backend plumbing, shared by SFT,
semantic adaptation and RL rather than implemented separately in each trainer.
Dense CUDA expert routing retains its fixed-row FP32 router independently of
expert dispatch, so selecting grouped FP8 does not also change top-k arithmetic.

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

Independent full-size prefix qualification uses complete adjacent admitted train
examples and the unchanged production `packed_graph_loss`. Diagnostic selection
chooses a small useful whole batch within the existing 32k packed-token policy;
no target is truncated or reweighted. Crisp code records immutable source/model
identities, exact original-to-shared physical maps, gathered hidden/logit/loss
comparisons and summed input gradients through every full-size backbone layer.
Parameters are frozen only to avoid allocating full-model gradient storage; a
leaf embedding hook retains the actual backward path. A nonzero producer adapter
checks that ordinary cached prefixes remain unadapted. RNG/backend state is
restored between variants. Mellum native BF16 and production FP8 expert modes
are distinct required cases. Timing under a concurrent held collector is
telemetry, not an isolated throughput claim. A failed numerical comparison stays
failed until localized and repaired; the diagnostic does not widen tolerances.

Gradient qualification also requires global relative L2 and cosine agreement;
small absolute gradients cannot qualify through an absolute tolerance alone.
FP8 qualification records executed expert projection calls, and rejects an
entered backend scope that ran no FP8 projections. Configuration metadata alone
is not execution proof. Numerical failures remain explicit in the report and
retain exact CPU tensors for the assigned localization.

### Layout-independent native attention accumulation

Equivalent cached and graph executions must visit each query's visible KV
blocks in the same chronological order. FlexAttention processes partial blocks
before full blocks; moving a KV block between those lists changes FP32 online
softmax accumulation and can cross BF16 routing boundaries. Dummy-query ranges
and query-tile origins change that classification even when every actual query
sees identical keys. The first Mellum layer reproduced this with bitwise equal
Q/K/V, and an all-masked chronological sparse traversal restored bitwise output.

1. Keep the existing exact per-query visibility bounds and sparse overlap census.
2. List every overlapping KV block once, in ascending position, in the masked
   traversal. Do not separately reorder fully visible blocks.
3. Execute the existing native sparse tensor-core forward/backward and fixed
   per-row delta; discard all attention-only dummy outputs as before.
4. Qualify full graph, cached and import-gap outputs/gradients for both ports,
   including actual FP8 execution, and measure the mask-fast-path tradeoff.

This is exact numerical mechanism, not data or promotion policy. No causal key,
loss target, row normalization, learned role or exposure credit changes. Full
block fast-path restoration requires an ordered kernel implementation with the
same numerical contract; changing a tolerance is not a repair.

### Explicit expert input-adjoint precision

The supervisor may trade training throughput for a more accurate adjoint after
measuring actual gradients, updates and task behavior. This is training policy,
not a reason to silently change an existing production backend. The current
FP8 input-gradient default remains unchanged. A declared `current-bf16-dx`
recipe selects the same grouped FP8 forward and BF16 weight-gradient mechanism,
with a BF16 tensor-core input adjoint through the forward's dequantized weights.

1. Bind the requested input-adjoint precision to backend configuration,
   checkpoints, inventory and activation recomputation.
2. Reuse the expert block table and chronological reduction over incoming
   gradient coordinates. Load the original BF16 incoming gradient; dequantize
   the saved forward FP8 weights into BF16 tiles; accumulate dot products in
   FP32 and return the existing BF16 adjoint shape. Do not quantize incoming
   gradients to E4M3 for this recipe or retain a full dequantized weight copy.
3. Keep forward routing, forward operands, weight gradients, parameters,
   optimizer moments, RNG and source-exposure obligations unchanged.
4. A declared stateless precision transition records its previous/requested
   modes. SFT continuation requires actual restored-model backend qualification;
   it cannot pass by changing checkpoint metadata alone.
5. Compare the exact FP64 dequantized-weight adjoint, executed kernel cost,
   checkpoint replay and actual optimizer updates before selecting a new
   production mode. Report BF16 fan-in and output rounding separately from the
   additional FP8 incoming-gradient surrogate.

The grouped multiply, state verifier and CLI binding are crisp numerical and
durability mechanisms. Recipe choice and adoption remain the supervisor's
adaptive training policy. Both students use the shared backend interface;
this option applies only to the architecture with grouped dense experts.

### Learned-row rates and named optimizer continuation

The supervisor may give the learned reasoning/Neuralese vocabulary rows a
separate declared learning rate when actual gradient/update and held-task
measurements justify it. The backbone and embedding adapter keep their declared
rates. Defaults remain unchanged; a line-specific measured choice is not a
claim of lift on the other student.

1. Identify learned vocabulary rows by their existing exact interface parameter
   names, including added reasoning rows. Apply rates with the shared
   `optim_restore.set_learning_rates`; never create a second optimizer.
2. Bind every serialized optimizer slot to its actual parameter name in child
   group order. Constructor schema order is a contract inventory, not slot order.
3. On stop, repack, group split, reorder, sparse-row addition or FP32 promotion,
   restore each existing moment and per-parameter group configuration by that
   name. Recover old one-group/explicitly named latent layouts only from their
   exact existing bindings; reject ambiguous historical split layouts.
4. Apply explicitly requested new rates after state restore, preserving moments,
   RNG, exposure and update counters. If no row-rate override is declared, retain
   an inherited distinct learned-row rate through SFT, semantic phase changes,
   behavioral distillation, RL and sharded checkpoint loading.
5. Record effective per-name rates and exact declared changes in checkpoints and
   progress. Expose the same optional setting through shared recipes and all
   native training stages, with architecture differences limited to measured
   chosen values.

Rate selection is the supervisor's measured adaptive policy. Slot identities,
state migration, group splitting, validation and recipe plumbing are crisp
mechanisms; they do not select a new training policy or discard old history.
