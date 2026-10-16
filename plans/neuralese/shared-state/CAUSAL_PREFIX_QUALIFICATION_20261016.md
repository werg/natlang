# Causal-prefix qualification: open

Production prefix sharing remains off. The independent review identified and
repaired a shared attention accumulation-order defect. Full-size LFM now passes;
Mellum forward passes with actual FP8 execution, but its gradient qualification
remains open.

Frozen production inputs were commit `5706256bc`, with all 426 tracked runtime
files verified unchanged in
`/home/werg/data/dgxops/native-prefix-code-inventory-20261015.json`.
The exact admitted V5 train graphs, selected adjacent row IDs/hashes, expanded
tokenizer fingerprints and actual pinned base inventories are bound by the
full-size reports. No checkpoint download, held training, target truncation,
loss reweighting or data admission occurred. Frozen model parameters avoid
full-backbone gradient allocation; actual leaf embeddings carry gradients
through all layers and unchanged `packed_graph_loss`.

Before repair:

- LFM QAD FP32, bf16x6 dense arithmetic: 12,581 original queries became 6,697
  physical queries (46.77% reduction). Hidden max error 8.82e-6, relative L2
  1.16e-6; input-gradient max error 7.63e-6, relative L2 3.89e-6. Loss matched
  bitwise. The initial blanket logit gate FAILED: maximum 1.34e-4, including
  near-zero vocabulary logits. Report:
  `/home/werg/data/dgxops/native-prefix-full-lfm-20261015.json`.
- Its largest logit gap was traced through actual final normalization and the
  effective vocabulary row. Raw hidden error 7.75e-7 became normalized error
  9.16e-5; the exact FP64 delta dot product was 1.3462744e-4, explaining the
  observed 1.3399124e-4 after independently measured readout roundoff. First
  failing hidden primitive was layer 2's attention residual, preceding its
  FFN. The first 32 rows were bitwise through every norm/FFN hook, ruling out
  the initial global FFN-shape explanation on those rows. Reports:
  `native-prefix-lfm-first-divergence-20261015.json` and
  `native-prefix-lfm-failed-row-divergence-20261015.json` in the same directory.
- Mellum BF16/invariant experts: 12,365 queries became 6,297 (49.07% reduction).
  Hidden relative L2 .02879; embedding-gradient relative L2 .22665 and cosine
  .9848154; maximum logit error 23.039. These are substantive failures.
  The nominal FP8 case in that first probe executed the same invariant path;
  it is NOT FP8 qualification. Root repaired backend dispatch separately in
  `631a39b2a`. Report:
  `/home/werg/data/dgxops/native-prefix-full-mellum-20261015.json`.

Every imported causal history matched its exact original map. Mellum first
failed in layer 0 attention output with all raw/rotated Q/K/V bitwise equal.
The defect is FlexAttention's partial-then-full KV traversal: padding and query
origins change block classification and therefore FP32 online-softmax order.
One first-layer real-weight fixture reproduced 14,532 differing elements out
of 50,647,040. Causal dummy ranges alone and logical alignment alone failed;
combined padding and, independently, a single chronological masked sparse
traversal both restored BITWISE attention output. The latter repairs the
shared mechanism without a prefix-only or import-layout workaround. Report:
`/home/werg/data/dgxops/native-prefix-attention-padding-variants-20261015.json`.

The repair retains exact visibility and native sparse tensor-core kernels,
using one ascending masked KV-block list. It removes the separate full-block
traversal. Global forward/backward/cache and performance qualification are
required before production adoption; no tolerance was widened.

Ledgered validation after the repair:

- `native-prefix-ordered-mask-tests-20261015`: 18 CPU tests passed, including
  actual optimizer updates, continuation/exposure and loss normalization.
- `native-prefix-ordered-mask-cuda-20261016`: 5 CUDA tests passed: both-port
  graph/cached BF16 checks, fixed-delta cached adjoints and a non-128 prefix-cut
  regression with bitwise output and strict consumer-to-source gradient checks.
- The standalone qualifier rejects tiny-gradient absolute-tolerance-only
  success through global relative-L2/cosine checks, and records actual executed
  FP8 projection calls. A configured backend with zero calls fails.

All timings so far include the active historical LFM held collector. They are
paired diagnostic timings, not isolated throughput or task-lift evidence.
Full-size repaired checks used the fresh isolated `631a39b2a` runtime plus the
ordered-mask repair, in NGC 26.09 (Torch 2.14/CUDA 13.4) on the local SM121 GPU.
This does not establish CoreWeave SM120/restored-parent qualification.

- `native-prefix-repaired-lfm-20261016`: PASS. All 12,581 gathered hidden rows,
  all 824,508,416 logits and production loss match bitwise. Input-gradient
  relative L2 is 3.7567e-6, cosine .99999999999295, maximum error 6.85e-6.
  The original 1e-5 absolute/1e-4 relative gates are unchanged. Physical work
  remains 6,697 rows (46.77% reduction). Peak allocated CUDA 4,177,625,600 bytes.
  Report: `/home/werg/data/dgxops/native-prefix-repaired-lfm-20261016.json`.
- `native-prefix-repaired-mellum-20261016`: forward PASS, gradient FAIL.
  Both invariant BF16 and actual FP8 produce bitwise hidden states, all
  1,215,578,420 logits and loss. FP8 executed 336 actual grouped projection
  calls. Input-gradient relative L2 is .0173762 (BF16) and .0690566 (FP8),
  cosine .9998591 and .9976384 respectively. These are unresolved failures;
  no tolerance change or production enablement follows. Physical rows remain
  12,365 to 6,297 (49.07% reduction); peak allocated CUDA 30,518,377,472 bytes.
  Report: `/home/werg/data/dgxops/native-prefix-repaired-mellum-20261016.json`.
- `native-attention-order-cost-20261016`: actual first-layer Q/K/V, five
  alternating paired CUDA-event samples per layout. Disjoint forward medians
  25.161 ms legacy / 20.998 ms ordered; backward 60.403 / 72.671 ms (+20.3%).
  Shared forward 7.272 / 6.848 ms; backward 33.051 / 33.712 ms (+2.0%).
  These noisy concurrent-collector primitive timings do not establish full
  trainer throughput. Report:
  `/home/werg/data/dgxops/native-attention-order-cost-20261016.json`.

Remaining work: localize and repair Mellum adjoint differences against exact
references; qualify full cached/import-gap gradients and actual cloud runtime;
independent integration review before enabling either production recipe.


## Accounted adjoint arithmetic and optional precision experiment

The full layer-by-layer backward trace reproduces both original gradient gaps.
Its first difference is the final loss-gather fan-in: the shared BF16 adjoint
matches the FP32 sum of disjoint adjoints cast once to BF16 BITWISE, in both
expert modes. Native relative L2 before that cast is .00177365. Subsequent
BF16 backward operators amplify the different incoming adjoint. This is an
identified arithmetic effect, not an unresolved causal-mask defect or a waived
numerical qualification. Full trace:
`/home/werg/data/dgxops/native-prefix-backward-localization-20261016.json`.

A separate exact FP64 oracle uses 64 actual saved branch adjoints and the
published layer-27/expert-0 down weight. Current FP8 dX differs from the exact
forward-dequantized-weight derivative by relative L2 .0251091; direct FP32 dX
is 1.17e-7. Quantized input-adjoint additivity differs by .0122337, versus
.00165465 from BF16 fan-in alone. This nonlinear gradient quantization also
applies to existing FP8 training. Report:
`/home/werg/data/dgxops/native-fp8-adjoint-reference-20261016.json`.

The explicit expert recipe `current-bf16-dx` retains the same FP8 forward and
BF16 weight-gradient path. It uses BF16 incoming adjoints and dequantizes saved
FP8 weights into BF16 tiles, with FP32 accumulation. The current production
default stays unchanged. Configuration, checkpoint state, activation replay and
inventory bind the selected mode. Generic backend restoration inherits the
saved mode; an explicit stateless change records the previous/requested mode.
SFT continuation requires actual restored-model backend qualification.

- `native-fp8-dx-final-20261016`: 24 focused CUDA/backend/glue tests passed,
  including idle/partial experts, exact FP64 multiplication/roundoff bounds,
  bitwise unchanged forward and dW, actual native-graph Muon updates, complete
  optimizer/backend/RNG restoration and bitwise next-update continuation.
- On the tiny real-port native communication fixture, three current-mode
  updates had loss 5.038557 → 5.010185 → 4.978134; BF16-dX had the same initial
  loss then 5.008615 → 4.981847. Both changed actual expert weights. These are
  mechanism/update checks, not full-size model-lift evidence.
- `native-fp8-dx-cost-20261016`: all 64 actual published layer-27 down experts,
  weight shape [64,2304,896], 98,304 routed rows. BF16-dX relative error to
  exact FP64 dequantized-weight dX is .0016773, versus .0251091 FP8 (~15x
  lower). Five alternating CUDA-event samples gave median 15.671 ms versus
  12.767 ms (+22.7% for this primitive), with substantial variance under
  concurrent GPU work. This is not full trainer throughput. Peak allocation
  was 1,605,541,888 bytes; no complete dequantized weight copy is retained.
  Report: `/home/werg/data/dgxops/native-fp8-dx-cost-20261016.json`.
- `native-fp8-dx-quick-20261016`: quick check passed.

Full-size restored-parent training/task behavior and practical throughput still
select the production precision. Neither this optional recipe nor accounted
BF16 arithmetic silently converts the failed strict Mellum prefix gate into a
pass. LFM's separately qualified shared implementation may proceed to its own
actual H100/restored-parent handoff independently.

## Actual trained-parent qualification support

The shared qualifier now accepts an exact `--checkpoint` and optional native
participant. It preserves every learned row, adapter and backbone tensor;
trained parents are never overwritten with a random diagnostic adapter.
It checks the declared native arithmetic against the saved configuration and
inherits the saved FP8 recipe/history and actual backend state. Base-only
qualification remains explicitly separate. Staged runtimes can supply their
exact `--code-revision` without requiring a mounted Git checkout. The timing
note describes concurrent diagnostic scope rather than asserting that an old
collector still exists.

Both-model exact trained-parent restore/config/source checks plus existing
prefix objective and CUDA mask regression tests passed22 cases in6.68s.
The frozen production prefix package and actual H100 trained-parent test remain
in progress; prefix reuse is still disabled in production.


## Actual trained-parent H100 qualification and source-index continuation

On October16 the frozen3042af96a386 production runtime was extended only with
prefix packing, stable invocation serialization and the ordered sparse mask,
then the exact-parent qualifier051b4cc88 (candidate8fc03eb2c250). The H100 run
restored immutable trained6620 rather than starting from base. It passed:
13,431 original physical rows →8,188 (-39.04%), bitwise hidden/logits/loss,
input-gradient maximum5.59e-8, relativeL2 4.14e-6, cosine.999999999991.
Peak allocated CUDA was4.35GB. The concurrent28.39/7.24-second timings are
not isolated trainer throughput. Actual proof is
`/home/werg/data/dgxops/lfm-trained-prefix-8fc-actual-parent-6620-proof-20261016.json`.
No production prefix adoption is implied by this diagnostic.

Source-index continuation399162917 avoids revalidating unchanged raw graphs
for this exact extension. Six existing validation-policy files must be
byte-identical; the entire compiler AST must equal the previous compiler after
removing only the exact default-false field, serializer and three guards that
cannot run for parsed source invocations. The unchanged raw parser cannot set
that field. Existing content/tokenizer/split/role keys and every metadata entry
are preserved. New source bytes remain cache misses. Fourteen focused tests
include real old/new loaders, a cache hit without source parsing, and changed
source rejection; independent review is complete. Frozen3042→8fc policy proof
passed. This is neither new data admission nor numerical/capacity qualification.

Shared qualification loading must remain usable with the exact older runtime
being compared. Prefix packing and its loss helpers are imported only when
that prefix qualification is actually selected; loading an unchanged parent or
computing attention diagnostics cannot require a newer optional prefix module.
This is dependency plumbing, not a compatibility substitute for model code.
Missing required prefix support still fails when that operation is requested.
