# Stable native source admission and index reuse

The supervisor decides when useful data or training changes warrant a handoff.
These crisp helpers verify exact immutable data and compile positional graph
structure; they do not choose curriculum, quality or scheduling policy.

1. `data/native_graph_schema.py` validates token IDs, weights, source identity,
   split and external-state bindings. `model/native_graph_plan.py` compiles the
   same positional alias graph used by inference and training.
2. `data/native_pair_schema.py` verifies authenticated communication provenance
   across model views. The semantic trainer reexports and uses this one parser.
3. `data/native_sources.py` owns immutable streaming admission, compact indexes,
   row identities and actual producer/consumer checks. Its cache key binds all
   these modules plus exact JSON/hash/index plumbing, source SHA, tokenizer,
   split, model view and role IDs. Changes to these verifiers invalidate indexes.
4. Optimizers, numerical kernels, evaluation and logging remain outside the
   admission dependency closure. Changing them cannot force a full validation
   pass over unchanged data. Old module imports remain compatibility reexports.
5. Historical cache keys that did not bind the new dependency closure are cold
   once. No hash whitelist migrates admission across an unbound schema change.
   Corrupt metadata, changed files and changed semantic rows remain rejected;
   every streamed optimizer input is parsed and checked against its indexed row.

This is a relocation of existing exact mechanisms, not a second compiler,
parser or training path. Model-visible rendering and supervision are unchanged.

## Interleaved production reads

The curriculum decides batch order; reading must not make that policy expensive.
The supervisor keeps the existing communication/priority interleave. Exact crisp
plumbing maintains a separate monotone read cursor for each visit and declared
communication/priority rank. Each cursor uses the existing graph-source parser,
row binding and immutable-file checks. A batch must contain one rank, and each
stream must be monotone. Return batches in the original curriculum order, with
the exact original indices; close every cursor on completion or interruption.

This fixes repeated full-source decompression at every communication/ordinary
switch without changing admission, cache policy, targets, gradients or exposure.
Verify actual compressed-file opens and exact yielded rows under interleaving,
priority changes, nested manifests, missing credits and early iterator closure.

The production sample at 22:47 UTC on October 15 showed LFM step 3304 with
about 1.43 TB of process character reads in 48 minutes; Mellum step 3060 had
about 285 GB in its 11-minute continuation. These counters include pipes and
are not disk-byte totals. The source-reader defect is independently reproduced:
the old interleaved request opens both compressed leaves repeatedly, while the
new reader opens each once, yields identical requested IDs and validates every
row with the existing binding. Five focused tests pass, including nested
manifests, resumed multi-visit credits, mutation rejection and early closure;
all 30 existing native trainer tests also pass. Production throughput must be
measured after handoff; a single 0% GPU sample alone is not utilization history.
