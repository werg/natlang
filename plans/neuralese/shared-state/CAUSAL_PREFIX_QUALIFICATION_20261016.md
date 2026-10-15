# Causal-prefix qualification: open

Production prefix sharing remains off. The independent review identified and
repaired a shared attention accumulation-order defect; full-size qualification
of the repair and actual FP8 execution remain pending.

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
Remaining work: full-size repaired LFM/Mellum and actual FP8 comparisons;
benchmark the masked-traversal cost; qualify full cached/import-gap gradients;
independent integration review before enabling either production recipe.
