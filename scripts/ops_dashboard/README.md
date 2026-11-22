# natlang ops: live dashboard and training-data viewer (Tendriller application)

A session-connected Tendriller application (core 0.3.0-rc.17, contract 3, events state model; released packages from the
`source-659a75ba469f` release, first-app kit layout). The page is signed UI; one
owned Chronolog session carries its commands, queries and live signals; a native participant on the DGX applies writes and
answers reads.

| Piece | Where | Role |
| --- | --- | --- |
| `export.py` | ledger unit `ops-dashboard-export-N` (0.3 GB) | every 30 s writes `latest.json`, `metrics.jsonl`, `machines.jsonl`, `events.jsonl` under `<data>/ops/dashboard/out/` (unchanged contract, see `CONTRACT.md` there) |
| `app/ops-peer.mjs` | ledger unit `ops-peer-N` (1.5 GB) | the native participant: owns the native device, coordinates writes, serves reads, applies edits, mirrors the inbox; browser pairing runs under `helpers/participant-lifecycle.mjs` (woken by the host's setup notifications) |
| `data_service.py` (+ `data_catalog.py`, `data_items.py`, `data_edits.py`) | child of ops-peer (JSON lines) | answers every read: ops state and the training-data catalog, items, runs; applies edits to the edit layers |
| `app/domain.mjs` | bound into the signed descriptor | the retained state: `proposeEdit`, `recordApplication`, `ask`, `answer`; queries `edits`, `inbox` |
| `app/ui/` | the signed publication | dashboard (`app.mjs`, `charts.mjs`, `model.mjs`), training data (`data.mjs`), session connection (`connect.mjs`, `page.mjs`; the starter's `connection.mjs` and `connect-view.mjs` unchanged) |
| data registry | read by `data_catalog.py` | every dataset, its producer, inputs, license and consumers: `natlang_neuralese.common.data_registry` (git layer `training/corpora` + the producer journal; `scripts/data_registry.py`) |

Private state (handle, signer, epoch key, descriptor, device store) lives in `~/.local/share/tendriller/natlang-ops-app/private/`
(`NATLANG_OPS_PRIVATE`), never in the repository; the verified release packages (SDK, tools, Chronolog session; `package.json` installs them from there) are in `.../natlang-ops-app/kit-rc17/`;
the built page in `.../natlang-ops-app/page/`. Node 24: `~/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin`.

## How data reaches the page

- **Reads** are on demand over the session's live signals (`app/ops-protocol.mjs`): the page sends `ops/q {id, q, input}`, the
  participant answers with ordered `ops/r` chunks of base64 gzip JSON (signals are capped at 16 KiB). `ops/tick` (every 10 s)
  names the exporter files that changed, so the page refetches only those. Nothing read is retained in the journal: snapshots
  every 30 s and GB-scale training data would otherwise grow the signed history without bound.
- **Retained facts** are small and human-made: edits and questions (declared commands), the participant's application records
  and the agent's answers. The page reads them with the `edits` and `inbox` queries (live views).
- Query names: `ops.tick|snapshot|metrics|machines|events`; `data.catalog|dataset|items|item|tokens|chain|task_refs|search|registry`.
  `edits.*` and `inbox.*` are native-only.

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

## The supervising agent's inbox

`inbox.sh` (list), `inbox.sh wait` (block until an unanswered ask exists; run in the background), `inbox.sh answer OP TEXT`
(declared `answer` command through the running participant), `inbox.sh handled OP`.

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

Layout preview with real data (no session; edits stay in memory): `python preview_server.py` under the ledger, then
`http://127.0.0.1:8765/`. Tests: `pytest tests/ops_dashboard` (the Python data service and exporter, plus `app/test/app.test.mjs`: the WebSocket close guard and the module graph against the installed packages).
