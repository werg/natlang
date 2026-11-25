# Observed workspace materialization parity

Inventory before edits: `scripts/neuralese_data/live_teacher_sources.py`
`semantic_goal_completed`, `observed_decision`, and `load_episode` produce the
native IR from source-hashed call artifacts. `ts-host/scripts/observed-workspace-evidence.mjs`
`verifyEntry`, `decisionEligibility`, and `verifyObservedWorkspaceEvidence`
independently compare that IR with the same artifacts. The CLI
`ts-host/scripts/materialize-native-teacher.mjs` invokes this disk verifier before
`ts-host/src/teacher/native-materializer.ts` produces learning rows. Training,
student-improvement, static curriculum and teacher-generation scripts call that
same CLI (repository-wide reference search); they have no separate comparison.
The serving engines in ENGINE_PARITY do not apply observed capture admission:
they supply recorded model exchanges, not these offline receipt comparisons.

Actual0056: Python accepts26 captures/393 decisions, Node accepts23 and rejects3.
Two fail root-entry comparison because Node requires successful execution exit;
one differs in decision admission. Existing Python separates independent whole
goal completion from execution exit. Node must make the same distinction while
preserving actual root identity/input/source, independently bound goal proof,
partial scopes, failed-action exclusions, reviewed decisions and incorrect-return
exclusions. A failed executor's trailing prose without executed actions stays
excluded. No captured calls, raw packets, manifests or IR are rewritten.

The same comparison inventory finds model exchange phase (reasoned readout
versus action) and redacted thinking normalization have drifted from the Python
producer. Cover these canonical representations without accepting changed
content or fabricated reasoning. Underlying model/provider failures are not
resolved by correcting offline reader parity.

The final row converter `ts-host/src/teacher/observed-workspace.ts` independently
repeats the successful-executor requirement and previously conflated a passed
receipt with whole-goal completion. It also needs the same outcome/exit split:
bound completed goals can supervise actually executed actions even if the
executor later errors; done invocations may retain independently admitted useful
work without claiming the whole goal completed. Unfinished prose-only decisions
remain excluded, and reasoned-readout free generation remains context, not an
action training target. The disk verifier checks actual whole-goal evidence
before this converter runs. Direct converter calls retain structural checks.

Actual verification: rebuilt Node host plus61 distinct focused tests pass (one
optional external-archive test skipped). The unchanged recovered26 actual0056
rows now materialize393 eligible training turn records, zero rows unused, zero
held authored actions; actual unit `vibe-0056-offline-materialize-20261125c`
completed in2.1seconds/296.1MiB. Original source artifacts were not rewritten.
