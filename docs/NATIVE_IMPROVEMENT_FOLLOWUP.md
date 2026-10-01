# Native improvement follow-up

This follow-up tests the language through real Luna optimization of Bonsai-executed programs. The optimizer uses ordinary runtime tools and one diagnosis-and-edit directory reducer. Exact authored helpers measure, reject, accept and select source; `folder.iterateOn` owns finite repetition. Apps may append system instructions. There are no optimizer-specific prompt replacements, tool filters or runtime compatibility modes.

The benchmark defines twelve programs across classification, structured extraction, pagination and multi-file reducers. Independently labelled cases and a hidden reference implementation establish development headroom. Each program compares the combined reducer, a direct rewrite and frozen separated planner/editor source on the same ordinary runtime. Optimizer and student requests are recorded separately; executor model-call reductions do not imply lower total optimization cost.

## Verified training and language refinements

- Seven complete optimization loops replay with matching source, state, validation quality, request count and disposition. Recorded optimizer and student responses are reused; replay performs no provider inference.
- The current training export contains 73 admitted turns from 28 root/child invocations. All 28 invocation contracts replay independently, including their exact results and folder effects.
- Successful intermediate file edits and terminal actions are retained. Rejected tool actions remain in later context rather than becoming positive targets. Child tasks carry their actual source root and input request; root continuations carry their actual iteration state.
- Sealed confirmation cases are excluded from replay and training fixtures. Incomplete provider recordings for approved-sum and recommendation-flags are quarantined.
- Compiler 6 gives eval-created inline semantic functions stable parent-qualified identities. Fresh tasks have reproducible seeds; independently generated nested functions remain distinct and recursion rejection stays intact.
- Direct child loading preserves inherited and locally shadowed type aliases, so the same callable can be invoked through its owner or directly.
- The application collection adapter injects the flat evaluator at its owning root and executes each supplied invocation. Generic collection has no improvement-specific branch. Adapter identity is included in provenance.

Artifacts: `runs/native-next-six-20261001/replay-training-v6/verification.json` and `training-current-v6/manifest.json`. Reproduce with `replay-followup-study.mjs CLEAN_STUDY REPLAY_DIR`, then `export-followup-training.mjs REPLAY_DIR TRAINING_DIR` from `ts-host/`.

Verification: all 730 tests, 22 conformance cases, application builds, browser type checks and actual Chromium smoke pass. The portable browser adaptation fixture was regenerated for compiler 6.

The older unused exporter was removed: its useful provenance and source-group metadata are retained by the verified exporter, while root-restart labels and lost terminal observations are eliminated.

## Experiments still in progress

The following table reports completed development arms at this checkpoint. Quality is the declared validation score; calls are measured student requests. These small reconstructed tasks establish executable improvement examples, not a general learning-gain claim. Incomplete or timed-out recordings remain visible in the study ledger.

| Program | Baseline quality / calls | Combined | Direct rewrite | Separated |
|---|---:|---:|---:|---:|
| approved-sum | 0 / 2 | 1 / 6 | 1 / 3 | 1 / 5 |
| assignment | 0 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| confirmed-events | 0 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| priority | 0 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| quantity | 0 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| recommendation | 1 / 5 | 1 / 4 | 1 / 5 | 1 / 3 |
| recommendation-flags | 0 / 4 | 1 / 3 | 1 / 3 | 1 / 2 |
| resolution | 0 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| urgent-ids | 0 / 2 | 1 / 2 | 0 / 2 | 1 / 2 |

Selected sources are frozen before a single fresh confirmation case per program. Six completed confirmations currently show four native quality gains and two ties. Direct rewriting has the same four gains; separated source has five. These are descriptive results with one case per program; the protocol corrects the three comparisons and makes no significant learning-gain claim. Confirmation failures are not used to edit candidates or generate training.

The first optimizer-on-optimizer attempt incurred 74 calls and ended before publishing a source/state pair: optional `undefined` diagnostic fields violated the JSON iteration-state contract. The application now omits absent diagnostics, with a regression check that nested feedback round-trips through JSON. The failed attempt remains recorded. The corrected attempt uses only the original assignment’s remaining 326 calls, 64 rollouts, 13 proposals and elapsed-time allowance, on independent target families. Its final result is pending.

Final development, confirmation and meta reports remain in their respective run directories. Features and verified training are delivered independently of whether the remaining empirical objectives succeed.
