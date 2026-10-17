# Profile useful native training without replacing its implementation

The supervisor chooses a short window of real production updates, reads the
existing kernel/host breakdown, and changes the next optimization accordingly.
Both students use this mechanism. Profiling never supplies a new loss, optimizer,
model, data schedule, precision setting or qualification result.

1. Launch the actual declared native SFT or preference entry point with its
   original arguments and numerical package. Bind this observer separately in
   the supervisor's launch receipt.
2. Count calls to the existing complete `native_training_update`. The selected
   window is relative to this invocation, explicitly distinct from checkpoint
   global steps. Skip initial updates when desired; retain real data/dispatch
   gaps between recorded updates.
3. Reuse `common.profiling.UpdateProfiler` and its trace analysis. Add only
   observational scopes around the existing forward and backward calls and
   reuse the optimizer's existing `optim/...` regions. Optimizer classes and
   framework step hooks stay intact. Invoke every original operation once,
   with unchanged arguments.
4. Restore all original functions immediately after the finite window. Continue
   training. Diagnostic failures disable observation; training failures propagate
   unchanged and never count as completed updates.
5. Record the observer's relative window and outcomes beside the existing trace.
   Use the evidence to choose improvements; a trace is not task or gradient
   qualification. Preserve optimizer/RNG/exposure through ordinary continuation.

These are supervisor instructions in natural language. The wrapper is crisp
CLI/instrumentation plumbing over the existing shared profiler. It adds no model
facing text. Local equivalence checks compare actual parameters, optimizer
history and RNG with observation disabled and enabled before cloud adoption.

## Passive startup phase observations (2026-10-17)

Policy is unchanged: use the existing real training entry and continue its useful
updates. The crisp observer records begin/end/error timestamps for checkpoint
loading, model construction, carried-progress preparation, source construction,
optimizer construction/state restore, and checkpoint saving. It forwards the
same arguments, returns the identical objects, preserves exceptions, and never
loads a second checkpoint or samples RNG. Startup hooks are removed at the first
real update; the save timer lasts through the declared invocation. Timing I/O
failures remain diagnostic. The existing finite GPU profiler is unchanged.
