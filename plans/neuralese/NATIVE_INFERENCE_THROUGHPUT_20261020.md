# Shared native inference throughput

The native actor still executes real positional KV/hybrid-state producers and
consumers. No projection fallback, alternate attention arithmetic, token sampling
law, prompt, optimizer or policy-update objective was introduced.

## Ready mechanisms

- `e97599138`: reuse immutable one-query flex block indices/counts; create fresh
  causal/window bounds and actual sequence lengths for every invocation. The
  finite metadata LRU retains no mutable session bounds. Ordinary no-grad
  inputs omit the exactly zero low-rank residual; gradient-enabled training
  retains explicit zero adapter gradients and its optimizer semantics.
- `f63c8b05b`: unadapted no-grad execution omits parameter discovery used only
  to choose an autograd carrier. Functional adapter substitution and actual
  gradient/VJP paths keep their complete original routing.
- Fixed linear inference uses a16-row tile for at most16 actual rows, preserving
  the same grouped dense kernel, K/N tiles, precision and accumulation. Training
  and adjoints retain64-row tiles. This removes work on padded rows; it does not
  change backbone precision or parameter/optimizer state.

These mechanisms are shared by LFM and Mellum. Root independently reviewed the
first two patches as clear; the linear tile awaits integrated review/adoption.
Cloud resource control remains with the existing sandbox owner. This document
is not evidence that a running cloud actor has adopted the patches.

## Actual trained LFM evidence

Retained step17214 actor checkpoint:
`30bdfe1bfc6d2880867161af125a7760e17e6684c3850638453b5f7477489506`.
Source measurement is the public typed-return48 writer-v3 capture, not protected
held data. The direct benchmark reuses its first request's messages with an
explicit new seed31415, temperature0.8,96-position diagnostic cap and no tools:
the historical capture did not retain tool schemas. It therefore measures a
**derived public-message request**, not the original complete harness workload
or task correctness. All outputs remain diagnostic, never teacher training data.

The actor was loaded once per probe, then baseline/candidate requests alternated.
Sampled token IDs, exact log probabilities and generated-input coordinates
matched. The mask/adapter patch's two timed pairs were mixed (40.14→39.16 and
42.35→44.42 decoded positions/s); no robust full-size gain is established there.

Isolating only the linear tile after warmup gave:

| Paired decode |64-row baseline |16-row candidate |
| --- | ---: | ---: |
| first |46.350/s |61.725/s |
| second |46.359/s |62.754/s |

This is about34% faster decode in this bounded same-actor benchmark. Prefill
remained about0.382 seconds. All six requests, including two warmups, retained
exact sample/log-probability/input parity. It is not a full-harness throughput,
held-quality, current-cloud-policy or RL improvement claim. Load cost was
60.4 seconds for the tile probe; ledger peak was6.6GiB. The process exited after
measurement instead of retaining an idle model.

A separate controlled full-size probe used a private producer prefix, actual
producer marker/transform, authenticated stored four-token native state, then an
ordinary consumer with the real imported positional state. Exported state ID,
raw layer tensors, embeddings and consumer logits matched the prior execution
bitwise. This validates the channel mechanism, not an environment reward or RL
update. Report records `exact-native-state` consumption.

Artifacts under `/home/werg/data/dgxops/native-throughput-20261020/`:

- `fullsize-report.json`: mask/adapter same-actor parity and mixed timing pairs.
- `fullsize-tile-report.json`: isolated tile result, exact checkpoint/source,
  fused-kernel SHA, actual device/precision, sampler and message binding.
- `fullsize-state-profile.json`: authenticated producer/import result and trace
  SHA; `lfm17214-native-trace.json` is the active performance investigation trace.
- `tile-probe.json`, `mellum-bf16-timing.json`: mechanism timing only.

## Exact checks and Mellum limits

Eight CUDA tests compare16/64 row forward values and explicit input adjoints
bitwise across IEEE, bf16x6, TF32x3, TF32, BF16-mode FP32 inputs and native BF16.
They include odd dimensions, tiny adapter/sparse rows,1024 projections, actual
Mellum2304/1792 hidden/projected dimensions and a64k readout. Both canonical ports
also compare a whole producer/two-consumer fan-out graph and every parameter
gradient bitwise, including Mellum BF16. No tolerance was widened.

Fourteen focused decode/cache tests cover CPU/CUDA, fresh session bounds,
127/128/129/255/256/257 block boundaries, sliding windows, private-producer span
isolation, real native imports and exact captured KV/hybrid state. They retain
ordinary training zero gradients and gradient-enabled carrier discovery. The
Mellum BF16 cached-state case compares against the old64-row execution too.
Existing real-port adapter cached-graph-gradient/restoration checks also pass.
All builds/tests/probes were ledgered; quick ratchets and TS checks pass. Root
owns the integrated broad suite; no duplicate full suite was launched here.

The BF16 canonical tiny Mellum benchmark preserves exact final logits and shows
variable timing gains. Full12B Mellum/cloud throughput has **not** been measured.
Current kernel qualification is on the local GB10; additional GPU architectures
can run the same focused kernel/state checks at a useful owner-controlled window
without downloading a47GB student checkpoint. Training remains unaffected.

## Remaining measured work

The actual trained LFM profile attributes260.4/304.0ms GPU time (85.7%) over16
positions to invariant linears; flex attention uses35.5ms (11.7%). CPU overhead
includes linear dispatch, per-round Python work,112 stream synchronizations and
the seeded CPU multinomial. Preserve its distribution/RNG order when improving
sampling. These measurements justified reducing padded linear work first.

Potential next mechanisms, not implemented or required before adopting the
independently ready patches:

- Reduce inference-only linear AutogradFunction dispatch and unnecessary output
  zero-fill, preserving one shared forward and all storage/carrier contracts.
- Reduce one-token host-to-device/index construction where the profile confirms
  synchronization costs; preserve sparse-row identities and embedding behavior.
- Batch compatible native decode sessions through the existing scheduler, with
  real per-session positions, causal/window masks, hybrid state and captures.
  No continuous native batching is claimed by the current serial scheduler.
- Reuse exact immutable prompt-prefill state where valid, with concrete actor,
  adapter, token-plan and state ownership; never detach an in-episode producer
  from full-graph policy replay.

Template-prefix batching was investigated but withheld. CPU both-port checks
were bitwise exact; the first CUDA LFM stock FFN received identical inputs but
multi-row versus one-row GEMMs differed5.82e-10, propagating to a4.66e-9 hidden
state difference. That is localized unchanged stock-FFN shape arithmetic, not a
KV import corruption or model incapacity. The original sequential prefix stays
in place pending a separately justified numerical/runtime qualification; no
precision or tolerance was silently changed.

## Inference launch followup

The invariant linear now has one shared forward implementation. True inference
calls it without AutogradFunction save/backward bookkeeping while retaining
exact effective-storage casts. Ordinary no-grad and every gradient-enabled call
retain the Function. An explicit forward-only allocation hint skips zero-fill
only for fully written single-expert inference outputs. Grouped invalid/padded
expert rows and all adjoint/training allocations keep zeros, including an
explicit adjoint invoked inside an outer inference scope.

All32 focused checks pass: poison uninitialized output with NaNs, require every
cell finite and bitwise equal, cover all six precision/dtype modes and Mellum
2304/1792 geometry, preserve grouped padding and adjoint allocation, compare
actual adapted native producer/reader receipts on both ports, and retain every
prior fan-out parameter-gradient/cache/state check. Quick22/TS checks pass.

The same retained trained LFM derived-public-message benchmark passed exact
sample/log-probability/input parity in all six alternating requests. Timed pairs
were25.07→26.03 and18.84→24.99 decoded positions/s. Both improved, but shared
hardware load varied strongly; do not turn these into a universal speedup claim
or compare their absolute rates with the earlier row-tile window. Model load
was103 seconds and ledger peak8.9GiB; it exited after measurement. Evidence is
`fullsize-launch-report.json`. Four-way ctx/direct × zero/empty microprobes are
retained as `linear-split-probe.json`; they did not isolate a stable readout
regression. No full12B Mellum throughput or task-quality result is claimed.


## Routed expert inference followup

Dense routed experts now share the same inference row-tile selector as ordinary
linears. With at most 16 real routed pairs, true inference uses 16-row blocks;
ordinary no-grad, training and adjoint calls retain 64. Expert IDs, stable pair
ordering, routing weights, storage arithmetic, zero-padded invalid rows and FP32
weighted combination are unchanged.

Seven CUDA checks pass: F32/BF16 with and without activation clamping require
bitwise output and every input/expert/routing-weight gradient equality against
64 rows, including repeated expert contributions. Actual Mellum dimensions
(64 experts, hidden 2304, intermediate 1792, top 8) match exactly for one and two
tokens. Both canonical ports retain exact adapted producer/consumer captures.
The real gradient-enabled path remains 64 rows; the gradient check additionally
forces the smaller tile to validate the shared kernel arithmetic.

The synthetic expert-only probe is `moe-tile-probe.json`, without a 12B model
load. Its three one-token timing pairs (64→16 rows) were 1.54→2.59, 2.55→1.81
and 2.67→1.63 ms; two-token pairs were 4.90→2.88, 4.50→3.62 and 3.82→4.38 ms.
Shared-load variability remains substantial. These measurements establish
plausible saved padded work, not a universal speedup, full Mellum throughput,
cloud qualification or task-quality result. No cloud processes were changed.
The 32 existing shared CUDA checks and quick 22/TS checks also pass after this
shared-selector change (39 focused CUDA checks total). Root owns integrated
broad-suite validation; none was duplicated here.
