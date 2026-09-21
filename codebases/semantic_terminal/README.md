# Semantic terminal session

`reduce.nl` exposes the shared application signature and delegates to `step.nl`,
which reduces typed user requests, command completions and
cancellation requests into a session. Natlang chooses a recipe from the
session's catalogue and explains actual results. Crisp siblings launch native
jobs, correlate completion IDs, reject unverified completion events and retain
history. A request received while a job is active is explicitly declined for
resubmission; it is not silently queued. A cancellation request does not claim
the command was rolled back. The completion event records its actual outcome.

`RecipeTerminal` owns native recipe closures and job promises. It can bind the
build and media adapters in one retained TypeScript host without teaching the
natlang runtime about shell processes, FFmpeg assets or a global recipe menu.
The integration tests run both real adapters through an async Fold stream and
the same program through `TerminalNatlangApplication`. They check that native
job completions enter the event queue and that a forged completion cannot settle
the session. Host observations are included in the portable trace.

This is a trusted recipe terminal, not an unrestricted shell. The current
contract deliberately starts a recipe known to the host; it does not execute
model-authored shell text. Recipes may perform native effects that traces
cannot replay. `applications/semantic_terminal_cli.mjs` adds persistent
state/event IDs, per-reduction traces, Git/file/test/build workspace recipes,
actual abort propagation and explicit restart recovery for a lost job. `view.ts`
projects the semantic session into the shared terminal view schema. The
remaining product gates are live-model evaluation, richer composition across
build/media recipes, bounded incremental output and reconciliation for external
systems that can query operation IDs.
