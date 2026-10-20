# Shared initial raw contexts in complete communication graphs

`chooseUsefulWork(actualExposure, retainedShapes, qualifiedExecution) -> Continuation`
is the existing typed natural-language policy: keep full recipe/data obligations,
use shared mechanisms, and adopt a measured optimization after exact checks.
This change is crisp graph identity/state plumbing, not a new task policy or
model-facing instruction. The current cloud cutover remains independent.

Actual retained TRAIN pairs show long identical raw contexts before adaptation:
first128 foundation pairs have about70% possible prefix reuse in bounded-hash
65k batches; later Big2 pairs about54%. Even one paired graph has about19–32%
shared raw context. These bounded shapes are not population/speed measurements.
Full identical-producer dedup is not necessary to realize the main opportunity.

1. `identifyRawInitialContext(originalNode) -> Prefix` stops BEFORE the first
   actual import insertion or producing/logical-producing span. Tokens and
   positions before this event see no imported/adapted state. Token similarity
   after the event never aliases producer/consumer states.
2. `shareRawContexts(prefixes) -> RawCacheDAG` extends the existing exact trie and
   `cache_prefix`/complete logical-prefix imports. Empty prefixes remain empty.
   Existing ordinary-only plans retain their exact physical layout.
3. `retainEpisode(originalGraph, rawCaches) -> CompleteGraph` retains every
   producing suffix, consumer and edge, including fanout, repeated imports,
   empty-own-token state nodes and stop-gradient flags. Names are internal only.
   Each suffix imports its COMPLETE raw prefix at logical position zero, then
   retains all original event ordering. Raw caches never use the adapter.
4. `translateExactCoordinates(originalPlan) -> ExportsAndRows` converts original
   own-token export/import boundaries to exact original logical coordinates.
   This is essential when an import is at the split boundary; no prefix token
   can accidentally become a Neuralese payload. Preserve logical exports and
   every original own-token physical-row map, position and producing flag.
5. `superviseOriginalEpisodes(originalWeights, originalRowMaps) -> Loss` uses
   the existing mean of original graph losses, masks and multiplicities. Shared
   state receives all actual producer/consumer gradients. No source, exposure,
   optimizer, RNG or captured episode identity is rewritten or credited.
6. `verifyExecution(original, shared) -> Evidence` checks original histories,
   positions, flags, hidden/logit outputs, hybrid state and complete derivatives
   on both canonical ports; include multiple reads/fanout, zero masks, boundary
   imports and BF16 rounding. Unsupported precision must retain the original
   layout explicitly rather than receive broader tolerances.

External-state/policy-objective layouts keep their existing path; this CE
planner does not claim qualified RL occurrence/view routing. It is one shared
packing mechanism for both students, with no detached producer, recursive replay,
or partial consumer-only training. Future policy reuse needs its own complete
original occurrence-row mapping through the existing objective contract.

## BF16 primal reuse, original occurrence adjoints

Actual CUDA evidence localizes the BF16 discrepancy to activation fan-in, not
forward state: all mapped hidden/KV/hybrid values are bitwise equal, but merging
physical rows merges cotangents before independent BF16 casts. FP32 sparse
masters and FP32 parameter fan-in do not repair that activation rounding law.

1. Keep the ORIGINAL expanded graph, edges, physical rows and backward topology.
2. Derive a bound per-row primal equivalence proof from the existing raw-prefix
   planner; representative rows must belong to this exact original plan.
3. Carry that immutable proof through existing gradient-partition provenance.
   Token-local reordered/expert/chunk operands must not inherit unqualified row
   aliases. Bind the proof to one explicit forward frame, never a global cache.
4. For fixed native linear operators only, compute identical primal rows once
   and expand the result inside the existing custom forward. Save ORIGINAL
   operands/partitions; the existing backward executes every occurrence and
   original dtype cast, reduction and parameter contribution unchanged.
5. Verify exact donor outputs and every derivative on both production profiles.
   Expand operator coverage only after each original adjoint is preserved.

This first extension saves selected forward projection work, not activation
storage or backward work. It is not recursive replay or detached state; policy
selection and complete source/exposure requirements are unchanged.

The implemented BF16 path reuses fixed-linear CUDA primals only. FFN invocation
chunks, attention, finite-history convolutions and all custom backward bodies
remain original. Indices are allocated once per bound forward and reused during
checkpoint recomputation; no global/session GPU cache is retained. FP32 uses the
separately qualified physical DAG planner. Existing ordinary-only BF16 physical
prefix aliasing exhibits the same cotangent-rounding defect, so it also uses the
original-row primal path. Two actual shared-trainer updates on both BF16 ports
preserve every weight, optimizer tensor and CPU/CUDA RNG bit exactly, for both
ordinary and communication graphs. Telemetry distinguishes eliminated projection
rows from physical forward tokens and actual data exposure.
