# Retained collector artifact custody

The supervisor's natural-language recovery decision is: continue the same
request-bound partial collection only after preserving its complete communication
inputs, or retain the failure and do useful work elsewhere. This change implements
the exact custody mechanism; it does not choose retries, deadlines, tasks or models.

1. Before starting collectors, read each existing partial and authenticate every
   recorded evidence prefix. Identify content-addressed native artifacts in both.
2. Reuse the existing recursive RL bundle writer to preserve those exact artifacts
   and their dependencies. An existing bundle must name the same collection and
   policy. Missing inputs fail before any collector starts.
3. Restore that authenticated bundle into each exact-policy actor using the
   existing topological bundle reader and content-addressed HTTP PUT/GET. Preserve
   response bytes; do not regenerate or reinterpret an artifact.
4. After all owned collectors stop, preserve the latest partial closure while the
   actors are still available, including interrupted collection. The caller keeps
   its original exception if preservation also fails, with custody failure chained.

These are crisp helpers because hashes, dependency completeness, evidence-prefix
lengths and HTTP byte equality require exact verification. Existing collector
partial/request validation and fatal replay hydration remain authoritative. No
prompt, sampling, objective, output credit or production scheduling policy changes.
## Worker process custody before archival

Every collector worker starts an owned session. Reuse the existing engine
guard's exact birth/session verifier and cleanup, with an explicit session-only
scope for concurrent workers. This scope must never select the enclosing
process's other children or reap an unrelated worker. Drain the owned session
in a `finally` block after normal exit, cooperative interruption or any wait
exception, before the collection can archive partials. Keep the engine guard's
default whole-guard behavior unchanged. Exercise actual signal-ignoring child
and grandchild processes plus a concurrent unrelated sibling; mocks alone do
not establish process custody.

The worker verifies helper origin and session-only API before launch, writes its
exact helper SHA/size and command to a separate process-custody receipt, and
checks helper stability after drainage. This records the extra script dependency
without rewriting the existing sampling/request binding of retained partials.
Future sealed runtime compositions include this exact helper alongside the
native package; an old or cross-runtime helper fails before starting a worker.
