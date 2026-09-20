# Semantic terminal session

`step.nl` reduces a stream of typed user requests, command completions and
cancellation requests into a session. Natlang chooses a recipe from the
session's catalogue and explains actual results. Crisp siblings launch native
jobs, correlate completion IDs, reject unverified completion events and retain
history. A request received while a job is active is explicitly declined for
resubmission; it is not silently queued. A cancellation request does not claim
the command was rolled back. The completion event records its actual outcome.

`RecipeTerminal` owns native recipe closures and job promises. It can bind the
build and media adapters in one retained TypeScript host without teaching the
natlang runtime about shell processes, FFmpeg assets or a global recipe menu.
The integration test runs both real adapters through an async Fold stream and
checks that a forged completion cannot settle the session. Host observations
are included in the portable trace.

This is a trusted recipe terminal, not an unrestricted shell. The current
contract deliberately starts a recipe known to the host; it does not execute
model-authored shell text. Recipes may perform native effects that traces
cannot replay. Actual process interruption, Git recipes, a working-directory
profile, retained bounded log reads, session persistence and a live-model
pilot are the next product gates. These are host/library concerns unless a
concrete case demonstrates a missing natlang stream operation.
