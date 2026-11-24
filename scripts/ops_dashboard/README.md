# natlang ops: live dashboard and training-data viewer (Tendriller application)

A session-connected Tendriller application (core 0.3.0-rc.17, contract 3, events state model; released packages from the
`source-659a75ba469f` release, first-app kit layout). The page is signed UI; one
owned Chronolog session carries its commands, queries and live signals; a native participant on the DGX applies writes and
answers reads.

| Piece | Where | Role |
| --- | --- | --- |
| `export.py` + `ei_export.py` | ledger unit `ops-dashboard-export-N` (0.3 GB) | every 30 s writes `latest.json`, `metrics.jsonl`, `machines.jsonl`, `events.jsonl` under `<data>/ops/dashboard/out/` (see `CONTRACT.md` there). `ei_export.py` builds `snap.ei`, the expert-iteration section (below) |
| `app/ops-peer.mjs` | ledger unit `ops-peer-N` (1.5 GB) | the native participant: owns the native device, coordinates writes, serves reads, applies edits, mirrors the inbox; browser pairing runs under `helpers/participant-lifecycle.mjs` (woken by the host's setup notifications) |
| `data_service.py` (+ `data_catalog.py`, `data_items.py`, `data_edits.py`) | child of ops-peer (JSON lines) | answers every read: ops state and the training-data catalog, items, runs; applies edits to the edit layers |
| `app/domain.mjs` | bound into the signed descriptor | the retained state: `proposeEdit`, `recordApplication`, `ask`, `answer`; queries `edits`, `inbox` |
| `app/ui/` | the signed publication | dashboard shell (`app.mjs`: state, intake, alerts, ask dialog, tabs), views (`views.mjs`: Now, Rounds, Data, Held-out, Machines, Units, Teachers, Open items), charts (`charts.mjs`: line, stacked bars, round Gantt, held ladder), pure helpers (`model.mjs`), training data (`data.mjs`), session connection (`connect.mjs`, `page.mjs`; the starter's `connection.mjs` and `connect-view.mjs` unchanged) |
| data registry | read by `data_catalog.py` | every dataset, its producer, inputs, license and consumers: `natlang_neuralese.common.data_registry` (git layer `training/corpora` + the producer journal; `scripts/data_registry.py`) |

Private state (handle, signer, epoch key, descriptor, device store) lives in `~/.local/share/tendriller/natlang-ops-app/private/`
(`NATLANG_OPS_PRIVATE`), never in the repository; the verified release packages (SDK, tools, Chronolog session; `package.json` installs them from there) are in `.../natlang-ops-app/kit-rc17/`;
the built page in `.../natlang-ops-app/page/`. Node 24: `~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin`.

## What the page shows (phone first)

Six bottom tabs: **Now** (verdict and tiles, what needs attention, the training round and the one being prepared with their steps and ETAs, held progress, fresh data per hour,
machine cards, Colab credits and runway, what changed), **Rounds** (round timeline Gantt, policy versions, per round: steps, timings, what it trains on, new versus repeated
episodes, SFT and validation loss, collection health, frontier tiers), **Data** (episodes per hour by machine, per collector, per policy, student fill, teacher campaign per
teacher, campaigns by round, newest registry entries), **Held-out** (the gate, measurement in progress, pass rate by policy with its series, paired comparisons, every
measurement), **Machines** (DGX, Pop, Colab A100, Colab L4 with activity, memory, disks and their fall rate, credits and spend, watchdog, program tuning), **More** (Units grouped by role,
Teacher details, Training data, Open items and fallow, Investigations, Inbox, Events). Wide screens show every tab in the top bar.

`snap.ei` (built by `ei_export.py`, all readers incremental or cached by mtime and size; state in the exporter's `state.json` under `ei`, rebuilt when `STATE_VERSION` changes):
`driver`, `phase`, `lineage`, `reference`, `rounds` (stamps, size, SFT series, mix, frontier, health), `collection` (per policy, hourly episodes by machine, live collectors, fill),
`held` (results with series and task-level standard errors, same-series pairs, the gate, measurements in progress from Pop's held log, the held-v4 queue), `colab` (balance and rate
from `colab usage` every 10 min in a background thread, balance history, spend per session from `cycle-cost.json`), `watchdog` (active alerts and GPU idle from the ops watchdog's state), `hist`
(10-minute machine samples, 24 h), `disks` (fall rate), `tuning`, `registry`, `open_items`, `decisions`, `events` (curated from `drive.log`). Derived alerts (driver down, SFT log lag, credits runway,
disk fall, held gate retrying, stalled collector or teacher) join `snap.alerts`. The driver is discovered as the newest `ops/ei/*/drive.log`. Held pass rates are only comparable inside one measurement series;
the page and the exporter never pair across series.

## How data reaches the page

- **Reads** are on demand over the session's live signals (`app/ops-protocol.mjs`): the page sends `ops/q {id, q, input}`, the
  participant answers with ordered `ops/r` chunks of base64 gzip JSON (signals are capped at 16 KiB). `ops/tick` (every 10 s)
  names the exporter files that changed, so the page refetches only those. Nothing read is retained in the journal: snapshots
  every 30 s and GB-scale training data would otherwise grow the signed history without bound.
- **Retained facts** are small and human-made: edits and questions (declared commands), the participant's application records
  and the agent's answers. The page reads them with the `edits` and `inbox` queries (live views).
- The page reads only the snapshot (and the `ops.metrics` stream while the Investigations tab is open, for its old GRPO charts); machine history, events, rounds and held results travel inside the snapshot.
- Query names: `ops.tick` (exporter file signatures, plus `listener`: the agent's heartbeat) `|snapshot|metrics|machines|events`; `data.catalog|dataset|items|item|tokens|chain|task_refs|search|registry`.
  `edits.*` and `inbox.*` are native-only.

- **Freshness and the stale alert** (`ui/model.mjs` `feedState`). The page reads the snapshot right after connecting, on every return to the
  foreground and whenever no read succeeded for a minute, besides the ticks. A snapshot held from before (saved copy, or a phone that slept)
  shows "N old · refreshing" and raises nothing until a read finishes. Then: a read that fails while the data is over 150 s old raises
  "No connection to the DGX" (warn; the exporter may be fine); a read that succeeds but returns exporter data already over 150 s old raises
  "Exporter behind" (crit after 10 min).
- **Clock skew.** A Chronolog signal receiver rejects a frame signed more than `maxClockSkewMs` ahead of its own clock (default 2 s) and then
  tears down the whole peer connection; history (commands) still syncs between the reconnects, signals never pass. On 2026-11-24 the owner's
  phone drifted ahead of the DGX by ~0.1 s/h and crossed 2 s at about 09:50Z: from then on no query or telemetry reached the DGX and the
  connection flapped every 10-30 s (werg/tendriller#103). Both sessions now set `maxClockSkewMs` to Chronolog's maximum, 10 s
  (`SIGNAL_CLOCK_SKEW_MS` in `ops-protocol.mjs`). The page measures its clock against the signing time of the ticks, reports it as
  `clock_skew_ms` in its telemetry and shows a notice beyond 5 s (turn on automatic date and time on the device).

## Edits: from the page to the data

`proposeEdit {target{kind,dataset,id,ref,label}, action, text, correction?, field?, basedOn?}` is a signed operation. The participant
applies each one once (`data_edits.py`), then records `recordApplication {edit, status, detail, files}` (operation
`apply-<edit ref>`; the page trusts only records authored by the native device). Layers, all atomic and idempotent:

- `<data>/ops/data-edits/log.jsonl` (audit), `state.json` (flags, exclusions, annotations, corrections per item),
  `exclusions.jsonl`, `corrections.jsonl`;
- RL task exclusions also go to `<data>/ops/pool-errata/ops-dashboard.jsonl` (pool-errata format), which grpo and teacher_gen
  read at their next start or resume. Live training files are never modified.
- Held-out data is view-only: the domain refuses `held_task`/`held_episode` targets, and the applier refuses any held id or
  split group. SFT row, demo and episode exclusions are recorded but no builder reads them yet; corrections need
  re-qualification (see the findings doc).

## Mobile layout

Tendriller injects a fixed "Page options" button at the bottom right of every page (about 125 x 54 px, maximum z-index, not configurable; werg/tendriller#92). On a
phone the tab bar therefore sits under the header at the top, and the page keeps 100 px of bottom padding so nothing ends under the button.

## The supervising agent's inbox

`inbox.sh` (list), `inbox.sh wait` (block until an unanswered ask exists; run in the background, uncapped), `inbox.sh answer OP TEXT`
(declared `answer` command through the running participant), `inbox.sh handled OP`, `inbox.sh listener` (age of the heartbeat in seconds).

- **What the owner sees.** After Ask > Send the item itself shows "Sent HH:MM; waiting for the agent" at once (this device's pending command, kept in
  localStorage until the journal has it) and the agent's answer inline when it arrives, with a "New answer" chip and a "Got it" button. Unread answers badge the
  More tab and the Inbox entry; Now has an Inbox strip one tap from the top. The Inbox tab lists every question with its state and the listener card.
- **Is anyone listening?** `inbox.sh wait` touches `<data>/ops/dashboard/inbox.listener` every 10 s. The participant puts that mtime in every `ops.tick` (every 10 s)
  as `listener`; the page shows "agent listening" while it is under 3 minutes old and "nobody listening" otherwise, on the strip, on waiting questions and in the Inbox tab.
  A tick older than 2 minutes makes the state "unknown" (the participant is not reporting).
- **A restart of the participant (`ops-peer-N`) kills the agent's listener.** The old `wait_event.sh` (and the contract-3 collector) are gone; the listener is only `inbox.sh wait`.
  After every restart of ops-peer, and after every ask `wait` returns, re-arm it: `scripts/ops_dashboard/inbox.sh wait` as a background command. The heartbeat on the page tells
  you whether you did. The ask itself is always kept: it is in the journal and in `inbox.jsonl` whether or not anyone listens.
- Ask records carry no timestamp (the journal's `authorTimestampMs` is not in the `inbox` query result, and changing a query's read function changes the definition the
  descriptor binds), so the page keeps the send time of this device's own questions in localStorage; questions from elsewhere show "Sent earlier".

## Operate

    cd scripts/ops_dashboard/app && export PATH=~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin:$PATH
    memory_ledger.py run --unit ops-peer-N --budget-gb 1.5 --class service --priority medium --wait 120 --env OMP_NUM_THREADS=1 \
        --workdir $PWD -- $(which node) $PWD/ops-peer.mjs
    memory_ledger.py run --unit ops-dashboard-export-N --budget-gb 0.3 --class service --priority medium --wait 120 -- \
        /usr/bin/python3 ~/natlang/scripts/ops_dashboard/export.py
    npx tendril --session $PWD/session.mjs query edits null              # attach mode while ops-peer runs
    npx tendril --session $PWD/session.mjs command proposeEdit '{...}' --operation ID

`ops-peer.mjs` exits cleanly on SIGTERM (`helpers/cli-lifetime.mjs`). It imports `helpers/websocket-close-guard.mjs` first: Node (and
every browser) throws `InvalidAccessError: invalid code` when a client closes a WebSocket with 1011 or 1008, and Tendriller's
invitation listener does exactly that from its `error` event, so a host that refuses the connection (for example `No such tendril`)
used to kill the participant (ops-peer-16, 2026-11-22). The guard sends such a close as 1000; it is a no-op for valid codes.

Publish a UI change (one owner of the signer at a time: stop ops-peer first, restart it after):

    node build.mjs && npx tendril --session $PWD/session.mjs --mode own publish ~/.local/share/tendriller/natlang-ops-app/page --operation ops-release-N

**First connection (owner).** Opening the page connects; there are no buttons (the starter's connection component,
`app/ui/connection.mjs` + `connect-view.mjs`). A browser that is already registered connects at once, in any tab. A new browser
creates its own key and a pairing request; the ops participant stages the exact control change within seconds, and the page opens the
service origin once, where the owner taps **Approve this device**; the service returns to the page, which connects. Enrollment stays open
while ops-peer runs. A page that has not been claimed expires 24 h after creation: the owner opens the page URL and signs in once to claim it.

## Setting up the application (new page, new definition)

A new Tendriller application is needed whenever the host has dropped the old one (the 2026-11-21 fresh-storage cutover did: `No such
tendril`) or `domain.mjs` semantics change (the descriptor binds the definition; a UI publication does not replace it). Steps, with the
participant stopped:

    PRIV=~/.local/share/tendriller/natlang-ops-app
    mv $PRIV/private $PRIV/private-old                  # keep the retired identity out of the way, never in git
    cd scripts/ops_dashboard/app && export PATH=~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin:$PATH
    npm install --ignore-scripts                         # packages from $PRIV/kit-rc17/packages (see package.json)
    npm run create -- https://tendriller.dev OWNER_EMAIL "natlang ops"   # or the account-linked route: npm run setup:prepare / setup:finish
    npm run init
    node build.mjs
    npx tendril --session $PWD/session.mjs --mode own publish $PRIV/page --operation ops-release-1
    # then start ops-peer-N (Operate), then the owner opens the printed page URL, signs in to claim, and approves this browser

The release kit is the verified first-app archive of the Tendriller release the host advertises (`curl https://tendriller.dev/api/readiness`,
`build.sourceCommit`; archive and `SHA256SUMS` at `github.com/werg/tendriller/releases/download/source-<commit>/`). To move to a newer
release: download it, check `SHA256SUMS`, unpack `packages/` next to the old kit, update the three `file:` paths in `package.json`, run
`npm install --ignore-scripts`, and diff the kit's `session.mjs`, `worker.mjs`, `connection.mjs`, `pairing-*.mjs` and `helpers/` against `app/`
(ours differs only in names, the private directory, queue limits, the ops participant, and `edits` as the first query).

## Preview and tests

Layout preview with real data (no session; edits stay in memory): `preview_server.py` under the ledger (it needs `PYTHONPATH=training/neuralese:.` and the repo venv; `NATLANG_OPS_OUT` points it at a scratch exporter output, `--page` at a scratch build from `NATLANG_OPS_BUILD=DIR node build.mjs`), then
`http://127.0.0.1:8765/`. Tests: `pytest tests/ops_dashboard` (the Python data service, exporter and `ei_export`, plus `node --test` in `app/`: `test/app.test.mjs` the WebSocket close guard and the module graph against the installed packages, `test/model.test.mjs` the pure helpers).
