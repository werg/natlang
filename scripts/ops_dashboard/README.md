# natlang ops: live dashboard and training-data viewer (Tendriller application)

A Tendriller page (`@tendriller/core` 0.3.0-rc.17 from the `source-e87d62ac` release, Chronolog v2 `2e3df79f`, contract 3) with one
**agent** on the DGX. The page is signed UI that the owner's browsers open; the agent program (`tendril agent run`) is the
DGX side: it admits the owner's browsers, serves the live data, applies edits and keeps the questions inbox. Nothing about live
state is stored in the application: the exporter writes files, the agent serves them as live data feeds over the session's
signals, and only small human-made facts (edits, questions) are signed operations.

| Piece | Where | Role |
| --- | --- | --- |
| `export.py` + `ei_export.py` | unit `ops-dashboard-export-N` (0.3 GB) | every 30 s writes `latest.json` (+ `metrics.jsonl`, `machines.jsonl`, `events.jsonl`) under `<data>/ops/dashboard/out/` (`CONTRACT.md` there). `ei_export.py` builds `snap.ei`, the expert-iteration section (below) |
| `app/pages/ops/agent.mjs` | unit `ops-agent-N` (1.5 GB), `tendril agent run` | the page's native side: the starter's `openNativePage` (admission, publication control) plus live data feeds, edit application, a 10 s tick, page telemetry |
| `data_service.py` (+ `data_catalog.py`, `data_items.py`, `data_edits.py`) | child of the agent program (JSON lines) | answers every read: the snapshot and the training-data catalog, items, runs; applies edits to the edit layers |
| `app/pages/ops/inbox-worker.mjs` | unit `ops-inbox-N` (0.5 GB) | the questions inbox: receipts, `<dashboard>/inbox.jsonl`, presence. Attaches to the agent program |
| `units.sh` (+ keeper `ops-keeper-N`) | `scripts/ops_dashboard/` | start, stop, restart and status of the units; the keeper restarts what crashed |
| `app/pages/ops/domain.mjs` | pinned in the page's genesis | the retained state: `proposeEdit`, `recordApplication`, and the starter's agent-request pipe (`agentMessage`, `agentReceipt`) for questions; queries `edits`, `agentFeed` |
| `app/pages/ops/ui/` | the signed publication (`dist/`) | dashboard shell (`app.mjs`), views (`views.mjs`), charts, training data (`data.mjs`), pure helpers (`model.mjs`), session connection (`connect.mjs`, `page.mjs`) |
| `app/pages/ops/worker.mjs` | in the publication | the browser's session owner (a Worker): device identity, storage, session, admission, peer links |
| `app/shared/` | vendored from the starter kit, verbatim | the starter's maintained composition: native page, admission, publication, signal requests, data feeds, agent requests, connection component |
| data registry | read by `data_catalog.py` | every dataset, its producer, inputs, license and consumers: `natlang_neuralese.common.data_registry` |

The workspace is `scripts/ops_dashboard/app/` (the starter's agent-workspace layout: `tendriller.json`, `shared/`, `setup.mjs`,
`pages/ops/`). Its private state, `app/.tendriller/` (agent credential, the one shared signer, the wrap key, the page's device
store and genesis, publication records), is a symlink to `~/.local/share/tendriller/natlang-ops-app/workspace-private/`, outside
the repository and gitignored; never print or commit it. The verified release packages (SDK, tools, authority, session) that
`app/package.json` installs are in `~/.local/share/tendriller/natlang-ops-app/kit-e87d62ac/`. Node 24:
`~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin`.

## What the page shows (phone first)

Six bottom tabs: **Now** (verdict and tiles, what needs attention, the training round or live GRPO run with their steps and ETAs, held
progress against the fixed anchor, fresh data per hour, machine cards, Colab credits and runway, what changed), **Rounds** (round
timeline Gantt, policy versions, per round: steps, timings, what it trains on, new versus repeated episodes, SFT and validation loss,
collection health, frontier tiers), **Data** (episodes per hour by machine, per collector, per policy, student fill, teacher campaign
per teacher, campaigns by round, newest registry entries), **Held-out** (the gate, measurement in progress, pass rate by policy with
its series, paired comparisons, every measurement), **Machines** (DGX, Pop and, while they exist, Colab sessions with activity, memory,
disks and their fall rate, credits and spend, watchdog, program tuning), **More** (Units grouped by role, Teacher details, Training
data, Open items and fallow, Inbox, Events). Wide screens show every tab in the top bar. When the owner has stopped Colab
(`snap.ei.colab.state == 'stopped'`) the Colab machine cards disappear and the credits card says so.

`snap.ei` (built by `ei_export.py`, all readers incremental or cached by mtime and size; state in the exporter's `state.json` under `ei`,
rebuilt when `STATE_VERSION` changes):
`driver` (`expected`: the run is in the watchdog roster `ops/watchdog.json`; false while the aei1 driver is paused), `phase` (`waiting_on`,
`supply`: the driver's supply wait, `consecutive_neutral`), `lineage`, `reference` (the driver's moving reference), `anchor` (the run's start
policy, fixed), `rounds` (stamps, size, SFT series, mix, frontier, health), `collection` (per policy, hourly episodes by machine, live
collectors, fill), `held` (see below), `teacher` (per teacher: `status` configured/removed from `ops/ei/teachers.json`, `paid`,
`since_launch`), `colab` (balance and rate from `colab usage` every 10 min in a background thread, balance history, spend per session from
`cycle-cost.json`, `state` stopped/configured from the roster), `watchdog` (active alerts, `suppressed` alerts about resources the roster no
longer expects, GPU idle), `hist` (10-minute machine samples, 24 h), `disks` (fall rate), `tuning`, `registry`, `open_items` (the newest
`HANDOFF_*.md` next steps first, then `OPEN_ITEMS.md` in flight, then qualified free teachers), `decisions` (the 8 newest DECISIONS entries
with their cause/verdict and next-step paragraphs), `events` (curated from `drive.log`). Top level: `runs` (live GRPO runs read from their
`train.jsonl`), `fallow`, `queue` (units waiting in the memory ledger), `alerts` (derived alerts: driver down while expected, SFT log lag,
credits runway, disk fall, held gate retrying, no policy beats the anchor after neutral rounds, stalled collector or teacher, join
`snap.alerts`). The snapshot schema version is `v` (4; the page refuses another: `SNAPSHOT_V` in `ui/model.mjs`, checked against `export.py`
by a test). The driver is discovered as the newest `ops/ei/*/drive.log`.

**Held pass rates** count every planned sample (tasks x samples): `held.results[*].rate` is successes over counted samples as `held_eval`
computes it (an unscored or timed-out sample is a failure; only a sample lost to infrastructure is left out, and the held tooling re-runs
those), `rate_all` is over every planned sample, `unscored` and `infra` say what was missing. A policy is compared with the fixed anchor (the
start policy, `held.anchor`), not only with the previous round: `gate.vs_anchor`, `gate.verdict`, `held.no_gain` ("3 consecutive neutral
rounds and no policy beats the anchor"). Pairs may cross measurement series, as `held_eval compare` pairs them; each pair carries
`same_series` and the page says "across series". Results of different held sets (`set`), sampling laws or engine settings never pair.

## How data reaches the page

- **Live data feeds** (`app/pages/ops/feeds.mjs`; the starter's `serveDataFeeds` in the agent program, `useDataFeed` and
  `createSignalRequests` in the page; signals underneath). Feed `snapshot` is the exporter's `latest.json`: the agent re-reads it every
  10 s while a peer is connected, pushes it when it changed (a snapshot is larger than one signal, so the push is a notice and the page
  pulls) and answers the first pull after connecting and every new peer link. Feeds `data.catalog`, `data.dataset`, `data.registry`,
  `data.search`, `data.items`, `data.item`, `data.chain`, `data.tokens`, `data.task_refs` are on demand (`{query: input}`), never pushed
  or cached. A reply is one JSON value of at most 4 MiB, split into signals by the helper; JSON over 8000 characters is gzipped and
  base64'd first (`pack`/`unpack`). Nothing read is retained in the journal: 30 s snapshots and GB-scale training data would grow the
  signed history without bound. `test/app.test.mjs` checks that the feed list equals the `data.*` handlers of the Python service.
- **Retained facts** are small and human-made: edits and their application records (`ops/edit`, `ops/applied`) and questions and answers
  (`agent/message`, `agent/receipt`). The page reads them as live views (`createLiveView`): `edits` and `agentFeed`.
- **Tick.** `ops/tick` every 10 s from the agent program: the page build stamp (a tab pinned to an older publication moves to the selected
  release) and the supervising agent's inbox heartbeat. Its signing time gives the page's clock skew.
- **Freshness and the stale alert** (`ui/model.mjs` `feedState`). The page reads the snapshot right after connecting, on every return to the
  foreground and whenever no read succeeded for a minute. A snapshot held from before (saved copy, or a phone that slept) shows "N old ·
  refreshing" and raises nothing until a read finishes. Then: a read that fails while the data is over 150 s old raises "No connection to
  the DGX" (warn; the exporter may be fine); a read that succeeds but returns exporter data already over 150 s old raises "Exporter behind"
  (crit after 10 min).
- **Clock skew.** A Chronolog signal receiver drops a frame signed more than 2 s ahead of its own clock (in the previous release it then tore
  down the whole peer connection: the 2026-11-24 flapping, a phone ~0.1 s/h ahead crossed 2 s; werg/tendriller#103). This release does not
  let a session configure the limit (`maxClockSkewMs` is not passed by `createApplicationSession`; `SIGNAL_CLOCK_SKEW_MS` in `feeds.mjs`
  documents the 2 s), so the page measures its clock against the tick, reports it as `clock_skew_ms` in its telemetry and shows a notice
  beyond 1.5 s (turn on automatic date and time on the device). Commands (edits, questions) are durable operations and sync across the skew;
  only live data stops.
- **Page telemetry.** `ops/t` every 10 s from each page (counts, latest events, build, clock skew): logged by the agent program and kept in
  `<dashboard>/page-telemetry.jsonl`.

## Edits: from the page to the data

`proposeEdit {target{kind,dataset,id,ref,label}, action, text, correction?, field?, basedOn?, supersedes?}` is a signed operation. The agent
program applies each one once (`data_edits.py`), then records `recordApplication {edit, status, detail, files}` (operation id derived from
the edit's operation id, so a restart never duplicates; the page trusts only records authored by the agent's key, `configuration.agentKey`).
Layers, all atomic and idempotent:

- `<data>/ops/data-edits/log.jsonl` (audit), `state.json` (flags, exclusions, annotations, corrections per item),
  `exclusions.jsonl`, `corrections.jsonl`;
- RL task exclusions also go to `<data>/ops/pool-errata/ops-dashboard.jsonl` (pool-errata format), which grpo and teacher_gen read at their
  next start or resume. Live training files are never modified.
- Held-out data is view-only: the domain refuses `held_task`/`held_episode` targets, and the applier refuses any held id or split group.
  SFT row, demo and episode exclusions are recorded but no builder reads them yet; corrections need re-qualification.

## The supervising agent's inbox

A question from the page (Ask on an item, a stop request, an alert acknowledgement) is an `agentMessage` on the chat channel whose `data` is
`{kind: ask|request_stop|ack_alert, subject: {kind, id, title}}`; an answer is the supervising agent's reply message with `inReplyTo`.

- **Intake: the inbox worker** (`ops-inbox-N`, `pages/ops/inbox-worker.mjs`). It attaches to the page owner inside the agent program (never
  a second signer) and, from each coherent read of the feed, gives every question a `seen` receipt (`done` once answered, at once for an
  acknowledgement), mirrors the questions with their answers to `<dashboard>/inbox.jsonl` and announces its presence (a card bound to the
  process; the page shows whether the question is taken). It never answers and starts no conversation. It exits when the agent program goes
  away; the keeper starts it again. A question sent while it is down stays in the journal and is recorded when it returns.
- **The supervising agent** reads and answers with `inbox.sh` (every read touches `<dashboard>/inbox.listener`, which the tick carries to
  the page as "agent reading"): `inbox.sh` (open questions as JSON lines), `inbox.sh wait` (block until one exists; for a background command
  in a session, optional now), `inbox.sh answer OP TEXT` (a reply message through the running agent program, `tendril` attach),
  `inbox.sh handled OP`, `inbox.sh listener` (age of the last read). Check the inbox at every heartbeat.
- **What the owner sees.** After Ask > Send the item shows "Sent HH:MM; waiting for the agent" at once (this device's pending command, kept
  in localStorage until the journal has it), then the answer inline with a "New answer" chip and a "Got it" button. Unread answers badge the
  More tab; Now has an Inbox strip. The Inbox tab lists every question with its state and "Is the question taken?": **agent reading** (worker
  up and the supervising agent read the inbox in the last 3 minutes), **recorded** (worker up, the agent has not looked lately: it answers at
  its next check), **no inbox worker**, or unknown (no tick from the DGX).
- There is no automatic answerer. An unattended answerer would be a headless model run on the DGX triggered from a phone; that is a decision
  for the owner, not part of this application.

## Operate

    scripts/ops_dashboard/units.sh status               # agent, inbox, export, keeper units
    scripts/ops_dashboard/units.sh start                # whatever is not running (agent first, then inbox, export)
    scripts/ops_dashboard/units.sh keep                 # the keeper: every 30 s start what is missing
    scripts/ops_dashboard/units.sh restart [agent|inbox|export|all]   # stop (keeper first), start under fresh names
    scripts/ops_dashboard/units.sh stop [role|all]

The units are memory-ledger services with names `<role>-<epoch>` (a restart takes a fresh name). Stopping the agent program loses nothing:
data and questions are in the journal and in files; a restart reopens the page from its device store and the browsers reconnect. Stop the
keeper before taking a unit down on purpose (`units.sh stop` does). Logs: `journalctl --user -u ops-agent-N` (JSON lines: `{hub}`,
`{page, state}`, `{page, admission}`, `{page, connection}`, `{edit...}`, `{page: {telemetry}}`), `-u ops-inbox-N`.

CLI against the running agent program (attach mode, never a second signer), from `app/`:

    export PATH=~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin:$PATH NODE_NO_WARNINGS=1
    node node_modules/.bin/tendril status                       # workspace, agent, grants, broadcasts
    node node_modules/.bin/tendril --page ops self | doctor | status | publication
    node node_modules/.bin/tendril --page ops query edits null
    node node_modules/.bin/tendril page devices ops             # admitted devices, pending sign-ins, mode

Publish a UI change (an immutable release; the host keeps the newest three): `cd app && npm run deploy -- ops`. A change to `domain.mjs`
(reducers, topics) changes the genesis the page pins: it needs a new page (below), not a deploy. Open tabs move to the new release
themselves (the tick announces the build stamp).

**First connection (owner).** The owner opens the page URL (the `page_url` in `app/.tendriller/pages/<id>/page.json`; it contains a secret
path, never commit or paste it), signs in to tendriller.dev with the account the agent was registered for, which claims the agent and the
page; the browser creates its device key, the host attests it, and the agent program admits it and wraps the keys; the page connects by
itself (the connection card says "Device attached; waiting for admission" until then). The agent program must be running. An unclaimed page
expires 24 h after creation. More devices: sign in the same way (owner-account devices are administrators); `tendril page devices ops`,
`page approve`, `page remove-device`, `page mode ops strict` manage them, and the page lists them (`shared/helpers/devices-view.mjs`).

## Setting up the application (new page)

Needed when the host drops the page, `domain.mjs` semantics change, or the identity must be recreated. With the units stopped:

    cd scripts/ops_dashboard/app && export PATH=~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin:$PATH
    # first time only: the agent (one credential, one signer, one wrap key for every page); needs the owner's verified email
    npx tendril workspace init --service https://tendriller.dev --owner OWNER_EMAIL --handle natlang-ops
    npx tendril page new ops --name "natlang ops"     # creates pages/ops/ (it must be empty: move the source aside and back) and the page
    npm run init -- ops        # prepare the host's publisher policy; create genesis (reducers, permissions, keys) and the epoch key
    npm run deploy -- ops      # build and publish
    scripts/ops_dashboard/units.sh start && scripts/ops_dashboard/units.sh keep   # then give the owner the page URL (First connection)

`workspace init` needs an empty folder without `pages/`; move `pages/` and `shared/` away and back (as done on 2026-11-27). Never create another
agent to retry a lost reply; losing `.tendriller/agent/` loses the agent (a move to another place: `tendril checkpoint export FILE` with
the agent program stopped). To replace only the page, `page new` another slug in the same workspace and remove the old one
(`tendril --page ops remove --yes`) once the owner has the new URL.

## Moving to a newer Tendriller release

1. `curl https://tendriller.dev/api/readiness` (`build.sourceCommit`), `tendril changelog --since YYYY-MM-DD`, `tendril broadcasts`; read the
   `breaking` entries.
2. Download `tendriller-first-app-<version>.tgz`, `manifest.json` and `SHA256SUMS` from
   `github.com/werg/tendriller/releases/download/source-<sha>/`, check the sums, unpack next to the old kit
   (`~/.local/share/tendriller/natlang-ops-app/kit-<sha12>/`), `sha256sum --check packages/SHA256SUMS`.
3. Point the four `file:` paths of `app/package.json` (sdk, tools, authority, session) at its `packages/`, `npm install --ignore-scripts`
   under the ledger, and diff the kit's `shared/` and `setup.mjs` against `app/shared/` and `app/setup.mjs` (ours differs only in that
   `setup.mjs` has no page templates). Re-run the tests; `npm run deploy -- ops`; `units.sh restart`.
4. The rc.17 (`source-659a75ba469f`) to `source-e87d62ac` move (2026-11-27) was a Chronolog v2 cutover: pre-v2 pages and device stores cannot be
   migrated, so the old page (`omcpbc67mm72`) and identity were retired and everything was created again. What changed: the agent workspace
   replaced the per-application native participant (`ops-peer.mjs`, `session.mjs`, the pairing and host-adoption scripts,
   `participant-lifecycle` use and the WebSocket close guard are gone; `tendril agent run` with `shared/native-page.mjs` does admission and
   lifetime); browsers are admitted by the agent program from host attestations (no pairing flow, no approval screen); reducers are ES modules
   pinned in the genesis and run in QuickJS (commands return topic updates, `readEvents` reads a topic's events; no `program(emit(...))`);
   retained questions use the starter's agent-request pipe instead of a custom `ask`/`answer` pair; live reads use the starter's data feeds
   instead of our own query/reply protocol (`ops-protocol.mjs`); `createLiveView(session, derive)` replaced `createLiveView(session, {query})`;
   `op.materialized()` replaced `op.applied()`; the browser session is opened with `bootstrapBrowser` and `createApplicationSession`.

## Preview and tests

- **Layout preview with real data** (no session; edits stay in memory): `cd app && node pages/ops/build.mjs --preview pages/ops/preview`, then
  `scripts/memory_ledger.py run --unit dash-preview --budget-gb 2 --class service --env PYTHONPATH=training/neuralese:. -- .venv-neuralese/bin/python scripts/ops_dashboard/preview_server.py`
  and `http://127.0.0.1:8765/` (`NATLANG_OPS_OUT` points it at a scratch exporter output).
- **Tests:** `pytest tests/ops_dashboard` under the ledger (the Python data service, exporters and `ei_export`, plus `node --test` in `app/`
  through `test_app.py`). In `app/test/`: `app.test.mjs` (domain and reducer, the agent program's edit application and data service child, the
  inbox worker, the page's ask/feed mapping, the snapshot version), `feeds.test.mjs` (the live data path over an in-memory signal pair),
  `model.test.mjs` (pure page helpers), `fake_service.py` (a stand-in for `data_service.py`, selected with `NATLANG_OPS_SERVICE`).
- **Local release host** (docs/LOCAL_HOST.md in the Tendriller repository; `npm run dev:host -- --release <sha>`): a sealed copy of the release on
  loopback with a synthetic owner, for the browser flow of a page (create, init, deploy, claim, connect). In this release the local host has no
  agent hub (the upgrade of `/api/agent/hub` answers 500: its Durable Object binding is missing), so the agent program cannot admit a browser
  there. Test a `--browser-first` page (the browser creates the genesis) for the page itself, and the data path with `feeds.test.mjs`. The
  host preload patches only `fetch`; `ws` clients also need `NODE_EXTRA_CA_CERTS` and a `tls.checkServerIdentity` override.

## Mobile layout

Tendriller injects a fixed "Page options" button at the bottom right of every page (about 125 x 54 px, maximum z-index, not configurable;
werg/tendriller#92). On a phone the tab bar therefore sits under the header at the top, and the page keeps 100 px of bottom padding so
nothing ends under the button.
