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

These mechanisms are shared by LFM and Mellum. Root independently reviewed all five inference patches as clear and integrated
them through `4e8704a83`; integrated checks remain the supervisor’s responsibility.
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


## Full-graph RL actor deployment

The existing `rl.production.configure_native` actor already executes
`serve_direct_checkpoint.py --native-state` → `load_native_engine` → the shared
`native_step`, adapter, attention and routed/linear kernels above. Both raw-state
and shared-semantic modes use this path. The learner remains the shared native
trainer/objective; the inference-only guards leave its gradients and replay
semantics unchanged. No additional serving engine, queue or trainer is required.

The cloud operator confirmed the imminent R14 training package is
`/work/native-runtime/native-shared-production-classified-v14-cached-reference-cpu-v1-20261019`
(inventory `b789522986f1045ac54afa4c800a089d32fa2e313e3def72c90209962b294034`).
It must remain frozen. R15 contains R14 plus eleven joint modules and deliberately
does **not** contain these inference optimizations. No optimized actor package
path is approved, staged or active yet, and there is no live RL policy owner.
Foundation/role release and actual cloud composition remain operator-owned.

Executable provision for that boundary:

1. The operator composes one immutable actor repository from the qualified joint
   semantics plus these reviewed inference patches and the runtime-binding
   followup. Stage this same package closure on each selected worker; do not
   change global provider `NATLANG_REPO`, `PYTHONPATH` or the R14 training tree.
2. Native `production configure` accepts `--actor-runtime` naming that actual
   existing repository. Raw/shared bootstrap model declarations accept the same
   `actor_runtime` path. Defaults retain the existing repository selection. The
   factory verifies the serving script and source inventory, writes the explicit
   actor command, and sets `NATLANG_REPO`/`PYTHONPATH` for that child only. Learner
   and seed commands, their provider environment and optimizer ownership remain
   unchanged. A provider command override must preserve these exact bindings.
3. Every newly loaded native actor attests a path-independent Python source
   digest using the existing resident `code_inventory`. New production configs
   bind the actual selected actor package; collection rejects a different or
   missing implementation receipt and rejects mixed-version replicas. The
   expected receipt and actual actor receipts persist in the existing collection
   binding/report. Legacy all-unattested actors remain explicitly legacy evidence
   and cannot satisfy a new bound configuration. Semantic-channel binding retains
   this implementation attestation.
4. Keep the checkpoint, participant, tokenizer/embedding identities, sparse-row
   precision, semantic adapter and history policy exact. Use the existing durable
   actor store and archive all imported/produced state before actor teardown.
   Sampling keeps the seeded CPU multinomial and true behavior log probabilities.
   Whole-episode admission/replay still covers all actual sampled/forced turns,
   producer occurrences, consumer imports and downstream outcomes. No detached
   producer, consumer-only update or communication-edge eligibility gate is added.
5. Use the first useful train-only collection as the operational check, retaining
   its actual checkpoint/package/runtime receipts and outcome/throughput report.
   Do not launch another generic qualification campaign in place of useful actor
   operation. The provider’s existing SDK worker runtime/entry-point checks still
   apply; a separate actor path does not authorize an unqualified provider launch.

Validation: 57 existing actor/collection/controller checks passed, including a
real CPU HTTP actor, native-state write, actor restart and exact archived bytes.
After adding the separate actor path/environment and bootstrap plumbing, 82
checks passed and one existing optional test skipped. These include mismatched
code/replica rejection, path-independent source hashing, separate actor package
selection, child-only environment and preserved learner/seed configuration.
These are local mechanism checks, not full-size cloud RL qualification.

## Historical task diagnostics and remaining work

The protected LFM17694 report records 0/16: 14 repetition, one model token-budget
exhaustion, one semantic mismatch, zero infrastructure errors. This used the old
5d085 TS runtime. It does not predict the running LFM18006 evaluation on corrected
02b. Only these aggregate failure classes were used here; no protected task
content or answer was copied into training.

The existing shared curriculum already starts at tractable train tasks, favors
mixed-success practice and verified full-trajectory seeding for all-fail groups,
and retains rotating breadth including solved/harder tasks. Existing rewards
score correct execution as one with or without communication; repetition and
model-caused exhaustion score zero, while infrastructure and host wall-clock
stops stay unscored. Keep independently authored short return/tool/read-write
train tasks as the first useful actor work; new model-facing scaffolds require
the existing live qualification before admission. Preserve full-graph updates
and all future comprehensive source exposure. No new prompt or termination rule
was adopted from these held failures.

The serial native scheduler, CPU sampling and per-token device/index construction
remain measured opportunities. Current evidence does not yet justify changing
the RNG law or adding mutable device caches. Deploy the independently ready
linear/expert changes into useful actors before another full actor reload or
architecture qualification. Full12B throughput remains unmeasured.


## Immutable actor plus future role-controller artifact

The local artifact is
`/home/werg/data/dgxops/native-actor-coordinator-composition-v6-20261020`, with
`construction.json`, 372 byte-bound source/control files and package inventory
`4395eecce55a29524f477956bc704910f37dda03c4dccc259b69ac899e1311cf`.
It is **not cloud-staged or active**. `compose_native_actor_runtime.py` constructs
it from the exact R15 construction receipt
`cfa3bbe8e17772b31f2c868741be27334c04678220208a31a868b5ee22815703`,
plus only the reviewed inference/runtime-declaration commits and strict command
precedence followups. Its receipt records full parent/source/patch identities.

Only eight package modules differ from R15. R15 predates optional deferred
gradient machinery in the later inference patch ancestors; importing all of main
would bring unrelated training changes. The builder instead copies eight exact
reviewed inference definitions/methods into the older fused module and checks
that R15's `_InvariantLinear.backward`, `_DenseExpertProjection` and
`invariant_weight_gradient` remain byte-identical. All other changes use ordinary
three-way git merging with explicit conflicts. There is no established R15 CUDA
bug: the initial ancestor-dependency suspicion was corrected after reading the
actual carrier references. No unrelated deferred-gradient machinery was imported.

The package-only R15 tree did not include the coordinator's executable scripts.
The composition includes the exact reviewed serving entry, queue runner and its
process-custody import closure (engine guard, guardian and batch completion).
Task cases, pinned base assets, Node runtime/npm/protection closure and existing
provider SDK environment remain separately declared dependencies; none is
replaced by a placeholder or a claim that source files are models.

Each role has `role-control/{role}/publishing-manifest.json` and
`control-material.json`. The existing publishing loader loads actual bound
external sources, preserves installed module origins, and checks every six-file
role authority. The future closures retain the exact reviewed `5778b3422` task-runtime continuation
helper already staged in R14 (the host source closure predates that supplement).
The final external raw-rounds override also contains the new
actor-runtime declaration; updating only an installed package would have left
this shadowed. These are **new future role maps**, not fallback authority for
old R9/R12/R14 source-host proofs. Historical owner evidence stays unchanged.

Future activation follows the existing verified later-runtime handoff at a
natural stage/quantum boundary. The actual R14 owner/spec stays untouched. A new
owner must really import this package and have a matching source plan, optimizer,
RNG, exposure and geometry; no old source plan is relabeled. The copied reviewed
controller retains those checks. Its existing binder is generalized only to an
explicit `config.model` and mandatory `config.actor_runtime`, keeping the same
actual-owner protocol and task-runtime checks. Review refs compare path/SHA
with optional byte verification; the existing controller repair-proof producer
retains its two-field own-source contract, with byte counts kept in frozen refs.

Executable preparation uses three existing-purpose commands:

- `compose_native_actor_runtime.py` builds a new physical tree, authenticating
  the base construction, both source-control closures and publisher/controller/
  binder sources. It refuses an existing output and grants no readiness.
- `verify_native_actor_runtime.py --composition <existing tree> --out <receipt>`
  verifies every source byte, imports both actual publishing closures in separate
  CPU processes, checks their six maps and active factory keyword, and verifies
  the collector custody import. It refuses checkpoint decoding and allocates no
  model. Its result is metadata evidence, not CUDA or behavioral qualification.
- `prepare_native_actor_role.py --composition <existing tree> --role lfm|mellum
  --inputs <real inputs JSON> --out <new binder-config.json>` binds the real
  `entry_root`, role bootstrap, actual root-review receipt, provider-readiness
  receipt, publication configuration and frozen refs. Optional `task_runtime`
  is retained. It requires the review to cover the actual composed controller
  and binder and accepts path/SHA references with or without redundant bytes.
  It creates no owner, learner, cloud process, checkpoint or readiness receipt.
  The operator launches the existing binder only after the approved handoff.

The input files above must be real; no runnable cloud binder-config is claimed
before the new owner/bootstrap/root-review/provider bindings exist. Current
provider-global imports and R14 training are never changed by the preparer.

An existing binder may retain learner-root strings in a provider actor command.
Explicit `actor_runtime` now rewrites only a recognized direct Python launcher
or env-assignment wrapper followed by Python (`-u`/`-B` permitted) and the exact
native serving script. Unknown custom executables, shell wrappers and Python
`-c` commands fail explicitly. The original argv/declaration and learner command
are preserved. Eight targeted tests and quick checks pass.

Actual composed-kernel validation preimported and asserted the physical package
and kernel path before pytest's conftest: 39 CUDA checks passed, with bitwise
outputs/gradients, both-port adapted positional KV/hybrid captures and actual
Mellum expert geometry. The first attempted run was misdirected by conftest and
is **not** accepted as composition evidence. The final tree adds only metadata/
command wiring and collector scripts; its kernel bytes match that qualified
composition, so no duplicate model/GPU qualification was run. Both actual
publishing closures passed metadata checks. Three role-material tests cover
actual-input preservation, two-field review bindings and wrong-review/source
tamper rejection. No full-size cloud RL or task-lift result is claimed.


The final artifact also includes the exact three reviewed utility CLIs from
`04bd86020` under `scripts/coreweave`, including the preparer's sibling imports.
An actual artifact-only smoke pins all utility/native package origins, verifies
all 372 files, and runs each CLI's `--help` from `/tmp`. It records no preparation
execution, root approval, model allocation, GPU run, cloud mutation or readiness.
The source host can run the preparer against its real absolute references.

The v6 artifact is an actor/coordinator **overlay**, not a complete SGD source
runtime. `scripts/coreweave/install_native_actor_layout.py` now physically clones
an explicitly authenticated full donor, replaces its entire Python package with
4395, retains inherited nonpackage dependencies and relocates both roles into
separate explicit production control roots. It rebuilds only JSON paths and
bound JSON hashes; numerical/controller Python bytes remain unchanged. Inherited
publication dependencies are separately hash-bound before and after the clone.
The installer uses the existing `scripts/clone_runtime_tree.py` physical-copy
mechanism; ship that companion alongside the installer, and use v6's reviewed
compose/verifier utilities and package on PYTHONPATH. No broad whole-donor
content inventory is claimed from a package-only inventory.

A realistic local layout using the exact R15 package and actual captured cloud
publication helper (`ae71ebb6…cc56`, 21,235 bytes; namespace initializer
`a6b68461…2bd5`, 58 bytes) passed both relocated publishing closures and the
actual checkpoint publication-selection/validation/trainer-parameter APIs.
The captured helpers match both current LF R9 and M R15 full donors. This proves
the new integration against those exact sources, not actual cloud full-tree
staging. The first 2 GiB import probe exceeded its cgroup and earned no pass;
the successful 4 GiB probe peaked at 3.2 GiB. Three focused layout tests and quick
checks passed. No model/GPU qualification was repeated.

On the source host, use `--donor` with that host's real complete donor and its
`--donor-code`; pass a JSON list of the two explicit donor helper bindings as
`--dependencies`. Select the accepted final runtime
`/work/native-runtime/native-shared-production-4395-actor-role-v6-20261020` and
explicit `--lfm-control`/`--mellum-control` roots below
`/work/native-production/control`. Run the existing preparer with `--composition`
pointing to this **installed full runtime**, actual reviewed `--inputs`, and
`--out <role-control-root>/binder-config.json`. The physical binder's `H` then
matches its config and eventual controller spec `out`. Actual bootstrap/provider
inputs and review must bind the relocated files, not the overlay's host paths.
No owner, readiness or source plan is synthesized: owner-bound `--preflight`
remains pending until the genuine later-runtime handoff exists.
