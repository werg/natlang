# Bounded preparation of immutable source metadata

Keep feeding usable data into the evolving recipe while training runs. Verify
source bytes, split, tokenizer, roles, row identity, source groups and scheduling
priority exactly as before. This is exact data plumbing, with no model policy
or new training admission decision.

1. One shared manifest initializer verifies each bound child, cycle, source
   identity and priority. The ordinary source loader and CPU preparation call
   this same mechanism with explicit child factories.
2. The preparation factory validates and retains each child once before loading
   the next. Existing unchanged roots reuse verified objects. No global class
   replacement and no mutation of a running learner or its frozen modules.
3. Source metadata is read-only. Equal group strings/containers share storage;
   nested manifests share entry dictionaries when their effective priority is
   unchanged. Exact values, order and canonical hashes remain unchanged.
4. Test against ordinary loading across nested manifests, explicit/inherited
   priorities, duplicate IDs, cycles and changed files. Verify retained object
   identity as well as equal values.
5. During learner adoption, seed the same factory with the actual loaded train
   and held trees. Reverify immutable bytes before retaining a referenced tree;
   build only new subtrees and parent metadata. Keep the pool local to one poll
   so rejected requests cannot accumulate trees. Existing append/protection,
   geometry, runtime and atomic commit checks still run before changing SGD.
   Object reuse is not a claim that full production adoption fits host memory;
   measure that separately on the target VM.
6. Preparation can explicitly withhold publication (`publish=False`, CLI
   `--no-publish`). It writes the same fully checked immutable request while
   leaving the learner's pending pointer untouched, including a preexisting
   request. This permits a memory measurement without accidentally triggering
   adoption in a runtime awaiting upgrade. Default preparation still publishes;
   reprepare against the current generation/runtime before actual adoption.

The immediate consumer is the SHA-bound external CoreWeave CPU preparer. It may
use the unchanged frozen leaf parser/index policy and explicitly staged metadata
modules. No numerical capacity, task quality or training exposure is inferred.
