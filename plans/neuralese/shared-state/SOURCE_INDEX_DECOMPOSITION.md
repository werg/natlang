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
