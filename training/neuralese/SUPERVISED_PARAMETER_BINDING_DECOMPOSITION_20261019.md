# Declared stage parameters control the actual trainer argv

The existing recipe is the training policy. A seed command supplies the exact
launcher and fallback arguments; it must not override a later declared stage
with stale duplicated flags. Actual LFM argv showed a declared task evaluator
followed by an inherited empty evaluator, disabling the intended boundary check.

1. For native shared SFT/preference launchers, remove inherited seed-command
   options only when the resolved stage explicitly supplies that parameter.
   Reuse the existing argv option-removal mechanism, including value/equal forms
   and boolean positive/negative switches. Preserve undeclared fallback options.
2. Retain one existing trainer-argument splice, or add it for a native launcher
   without one. Custom launchers keep their existing interface.
3. Let the existing stage serializer supply the declared typed value once. Keep
   input-source binding and objective selection in their existing mechanisms.
4. Verify actual expanded argv for task JSON/cadence, learned row rates, boolean
   switches and source roles, including inherited empty task values. Both models
   use the same continuation mechanism. Preserve sealed requests; deploy the
   correction through an explicit state-preserving continuation at a useful
   checkpoint boundary, not an in-place rewrite of an active request.

This is exact CLI plumbing for the existing declared policy. It does not change
the policy, any prompt, task input, loss or optimizer history.
