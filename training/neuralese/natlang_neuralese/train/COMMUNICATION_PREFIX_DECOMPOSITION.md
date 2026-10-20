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
