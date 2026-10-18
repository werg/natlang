# Actual backbone update observation

The production supervisor needs to distinguish a small learning rate from an
optimizer that does not change its parameters. Existing sparse-row observations
do not establish what happened to the backbone. Observation must preserve the
actual update, optimizer state, RNG, source credits, and task execution.

The policy is: observe the first useful update and then the explicitly requested
cadence; compare deterministic sampled coordinates before and after that same
update; report their gradient and actual displacement separately. These are
diagnostics, not readiness, quality, saved exposure, or whole-tensor norms.

1. The existing supervisor chooses the observation cadence and sample count.
   These are explicit CLI inputs, not a second training scheduler.
2. A crisp read-only helper selects evenly spaced logical coordinates from each
   backbone parameter in the actual named optimizer inventory. It gathers only
   those coordinates, including for noncontiguous tensors; it never clones the
   full backbone or consumes random numbers.
3. The original shared update executes once with its unchanged arguments. The
   helper reads those same coordinates and the remaining gradients afterwards.
4. A crisp reporter computes sampled displacement, changed-coordinate counts,
   relative magnitude, and gradient magnitude. Missing gradients and nonfinite
   samples remain explicit. Zero sampled displacement is diagnostic evidence,
   not proof that the entire tensor was unchanged.
5. The existing profiler wrapper installs this observer outside its finite GPU
   profiling window, so finite profiler restoration retains periodic observation.
   All wrappers are restored at exit. Observation failures disable diagnostics
   and leave training running; actual training exceptions propagate.

These units are exact instrumentation and filesystem plumbing. They introduce
no prompts, model calls, optimizer arithmetic, parameter writes, source-policy
changes, or model-facing behavior. Both students and the preference entry use
the same shared native update function and the same observer. The frozen
numerical package remains unchanged; supervisors bind the observer source.
