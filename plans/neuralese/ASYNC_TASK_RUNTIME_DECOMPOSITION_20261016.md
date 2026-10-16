# Complete task evaluation while development continues

`prepareTaskRuntime(source) -> boundRuntime` is exact durability plumbing. Task
selection, source protection and scoring continue to use their existing natural
language policies and exact verifiers. No new model-facing instructions.

1. Use the existing task runtime inventory to identify the built implementation,
   npm manifests and source-protection files actually used by the evaluator.
2. Copy those exact files into a reusable content-addressed local runtime before
   sealing an asynchronous request. Physically copy the runtime dependencies and
   optional vendor assets; no mutable source-file hardlinks.
3. Check the original and copied inventories before publishing the directory.
   A concurrent rebuild is a preparation error, never a mixed-runtime score.
4. Use the prepared runtime in the existing request/spec/provenance. Continue
   verifying its files before and after execution, including protection files.
   New requests can adopt new code immediately; active requests finish on their
   declared implementation without blocking development or training.

The failed LF4000 evaluation changed compiled native-learning-client JS/types and
build stamps under its live checkout during execution. Its result stays unscored.
Retain its case outputs for this assigned diagnosis; never relabel the failed
aggregate or use held tasks as generation inputs.
