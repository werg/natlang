# Teacher episode budget and retained native evidence

Mechanism inventory: `scripts/neuralese_data/vibestudio_campaign.py` supplies the dedicated Host collector's existing `--test-timeout-ms` option; `scripts/collect-live-workflow-teacher.mjs` forwards it only to the selected teacher cases; `src/cli/systemTestCommands.ts:718` records it in the run configuration; System Testing's `test-runner.ts:293` applies it per case. All eight template domains use this same runner path, so no domain program needs an artificial budget or prescribed route. Ordinary test defaults and existing model-call budgets stay unchanged.

The campaign now exposes `--teacher-timeout-ms`, default 3,600,000 ms (one bounded hour per teacher episode including native preparation), instead of hardcoding 600,000. Explicit lower/higher positive values are supported. The old pinned Host collector accepts that existing option, so current runtime/template artifacts do not need rebuilding. A separate script-only Host successor gives direct `natlang:teacher` invocation the same default.

Outer waits do not impose another episode/wave deadline: the campaign uses `subprocess.run` without a timeout; the Host collector uses `spawnSync` without a timeout and batches cases by concurrency; `systemTestCommands.ts:757` waits through `waitForRun`, whose terminal-state loop has no elapsed-time cutoff. The controller unit has `RuntimeMaxUSec=infinity`; its startup timeout is a service startup setting, not a running-wave deadline. Thus 64 cases with 8 workers may span eight batches without a second ten-minute wave cutoff. No concurrency setting is reduced.

Wave0034 retained evidence proves the three wrapperless cases executed:

| Case | Distinct completed Luna provider replies retained | Completed child calls | Seconds from Eval start to first model request |
|---|---:|---:|---:|
| check-special-use-subdomains | 28 | 8 | 323 |
| verify-and-correct-tutorial | 30 | 8 | 323 |
| visitor-label-pilot | 21 | 15 | 366 |

Each last pre-cancellation checkpoint waited on `ModelGateway.turn` only 2.4, 4.4, or 3.2 seconds before lifecycle retirement. The terminal error is `eval runtime generation was retired by a planned lifecycle transition` after the fixed test deadline. This establishes active inference followed by deadline cancellation; it does not establish completed whole goals. Roots have no final record hash. Cold library builds consumed more than half the old deadline. The separate external-module cache identity correction addresses repeated compilation; the collection budget must also accommodate complex execution.

Evidence was read from checksum-verified copies of the frozen native SQL and SHA-256-verified `NatlangRecordsDO.blobs` record bodies. The 79 distinct response IDs have completed provider messages identifying `openai-codex:gpt-6-luna`. This is a lower bound because active root/parent calls lack final records; estimated token cost is not a provider billing receipt. Original packets and archive remain untouched. Audit files: `/home/werg/vibe0034-native-boundary-audit/{inventory,eval-details,paid-calls,completed-provider-audit}.json` and each `*-native-records.json`.

Future campaign provenance records `teacher_started: null` when the wrapper or positive call evidence is missing. A positive root/call observation records true. Missing capture never asserts that no inference ran. Historical immutable receipts are preserved.
