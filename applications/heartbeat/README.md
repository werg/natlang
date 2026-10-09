# heartbeat: the hourly check-in as a natlang program

Design: [plans/HEARTBEAT_PROGRAM.md](../../plans/HEARTBEAT_PROGRAM.md) (review item P1). A machine-native timer runs one
cycle per hour, independent of any Claude session. A cycle reads the machine through read-only collectors, asks three
natural-language functions for judgments, checks every answer with an exact verifier, and records the report.

```sh
natlang run applications/heartbeat -- cycle [--stage shadow|advisory|auto] [--no-record]   # the default command
natlang run applications/heartbeat -- collect          # the evidence only, as JSON
natlang run applications/heartbeat -- report           # runs/heartbeat/latest.md
natlang run applications/heartbeat -- compare          # recorded proposals against what the agents did
natlang run applications/heartbeat -- record --action relaunch-run --target v20-train   # an agent action with no visible effect
```

Model options (`--profile`, `--provider`, `--model`) go to `natlang run` as for the specializer. The cycle calls the
executor (`settings.executor_endpoint`, `:8083`), waits for it to be idle like the specializer, and, when it does not
answer, produces the report with the crisp status ladder and says so.

## Parts

| Part | Decision | File |
| --- | --- | --- |
| reading units, logs, ledger, gates, mailbox, resources | crisp, read-only | `collect/` |
| minutes since progress, error candidates, gate fields | crisp | `collect/log.ts`, `collect/gate.ts` |
| is the run healthy, and why not | `diagnoseRun.nl` (typed `RunDiagnosis`) | `diagnoseRun.nl` |
| what the inbox asks of this machine | `triageInbox.nl` (typed `InboxTriage`) | `triageInbox.nl` |
| what to do now and next | `planNext.nl` (typed `Plan`) | `planNext.nl` |
| status ladder, message rules, cause lookup | crisp side of each part (`crisp` mode, shadow reference, degraded mode) | `ladder.ts` |
| quotes and ids occur in the readings, gate rule, proposal validity | exact verifiers; failures go back to the function as `feedback` | `verify.ts`, `actions.ts` |
| which actions exist, reversibility, owner, auto-apply | data the owner edits | `actions.json` |
| declared runs | data the owning session edits | `watch.json` |
| carrying out an allowed action, receipts | crisp applier | `apply.ts` |
| agreement with agent decisions | crisp comparator | `compare.ts` |
| records and `latest.md` | crisp | `record.ts` |

Each of the three decisions is a `pluggable` part (`settings.modes`): `nl` (default), `crisp`, or `shadow` (serve crisp,
record whether the natural-language side agrees). When a natural-language answer still fails its verifier after
`settings.repairs` repairs, or the model errors, the crisp side serves and the record says `crisp-fallback`.

## Stages

`settings.stage` in `actions.json`; the owner edits it. A command-line `--stage` can only hold a cycle back.

| Stage | Applied automatically |
| --- | --- |
| `shadow` | `record-heartbeat` only. Agents do not read the report. |
| `advisory` (shipped default) | `record-heartbeat` only. Agents read the report and record whether they followed it. |
| `auto` | also the actions marked `"auto": "promoted"`: `read-more`, `release-cache`, `schedule-recheck`, `set-status-page`. |

Never applied at any stage, whatever a file says (the loader holds them back and the record carries a warning):
process control (`relaunch-run`, `stop-unit`, `adopt-unit`, `queue-next`) and irreversible actions (`reply-note`,
`send-note`, `ack-inbox`). They are proposals to the owning session. Each applied action writes a receipt (probe before,
argv, exit code, probe after) into the record.

## Scheduling

`scripts/systemd/natlang-heartbeat.{service,timer}` (minute 17 of every hour, `Persistent=true`). The service runs
`scripts/heartbeat_cycle.sh`: a lock so cycles do not overlap, and a small claim through the memory ledger
(`memory_ledger.py run --budget-gb 0.5 --reserve-gb 0`). `scripts/install_heartbeat_timer.sh` copies the units;
enabling is process control and left to the machine's owning agent:

```sh
scripts/install_heartbeat_timer.sh            # copy and reload, timer stays disabled
systemctl --user start natlang-heartbeat.service   # try one cycle, then read runs/heartbeat/latest.md
systemctl --user enable --now natlang-heartbeat.timer
```

## Tests

`ts-host/test/heartbeat.test.mjs` runs the collectors, ladder, verifiers, applier, comparator and whole cycles on
fixtures (`ts-host/test/fixtures/heartbeat/`, a fixture host in `ts-host/test/support/heartbeat.mjs`) with scripted
models. No test loads a model.
