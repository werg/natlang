# Inherited communication-foundation readiness refresh

## Policy

The initial finite communication cohort remains exact once earned. Appending ordinary supervised rows does not expand that cohort or erase mandatory exposure. A later SFT continuation may publish a fresh foundation-readiness receipt either after it completes its current declared supervised exposure with at least one real optimizer update, or when its owner explicitly requests a current-weight readiness check at a stage boundary. The latter is used by RAW's existing post-RL foundation-current boundary; it does not wait for every ordinary debt row. The receipt binds the current checkpoint and passes the usual task/runtime checks. Partial refresh quanta and operational stops do not trigger the automatic path; this avoids repeating the expensive full checks at every checkpoint or SGD step.

## Functions and exact mechanisms

1. `communication_foundation` continues to verify the same declared finite cohort, row bindings, visits, and cohort digest.
2. The trainer distinguishes first completion from inherited refresh. First completion keeps its existing boundary. Automatic inherited refresh requires the exact prior cohort digest, complete current invocation exposure, at least one update beyond the restored step, non-policy operation, and no stop signal. The explicit stage-boundary request requires the same complete fixed cohort and no stop/policy operation, but may run with remaining ordinary debt.
3. The inherited refresh reuses the normal final task cadence and full runtime qualification on the current checkpoint. It changes no exposure counts, cohort membership, optimizer, RNG, or source binding.
4. `foundation_ready` is emitted only when that current-checkpoint runtime proof and task execution receipt both pass. Mandatory supervised obligations remain separately reported and remain incomplete until their own quotas finish.
5. A continuation regression starts from a checkpoint that already records the cohort as complete, appends ordinary rows, and verifies one fresh current-checkpoint foundation receipt at an explicit stage boundary while further ordinary debt remains. A non-explicit partial quantum remains unready.

## Implementation boundaries

Policy is the natural-language rule above. The trainer's exact boolean boundary is crisp because it gates an expensive verifier and produces a bound receipt. No new model-facing text, source transformation, optimizer behavior, or readiness exception is introduced.
