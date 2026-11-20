# natlang ops: live dashboard and training-data viewer (Tendriller application)

A session-connected Tendriller application (core 0.3.0-rc.16, contract 3, events state model). The page is signed UI; one
owned Chronolog session carries its commands, queries and live signals; a native participant on the DGX applies writes and
answers reads.

| Piece | Where | Role |
| --- | --- | --- |
| `export.py` | ledger unit `ops-dashboard-export-N` (0.3 GB) | every 30 s writes `latest.json`, `metrics.jsonl`, `machines.jsonl`, `events.jsonl` under `<data>/ops/dashboard/out/` (unchanged contract, see `CONTRACT.md` there) |
| `app/ops-peer.mjs` | ledger unit `ops-peer-N` (1.5 GB) | the native participant: owns the native device, coordinates writes, serves reads, applies edits, mirrors the inbox, reopens browser pairing windows |
| `data_service.py` (+ `data_catalog.py`, `data_items.py`, `data_edits.py`) | child of ops-peer (JSON lines) | answers every read: ops state and the training-data catalog, items, runs; applies edits to the edit layers |
| `app/domain.mjs` | bound into the signed descriptor | the retained state: `proposeEdit`, `recordApplication`, `ask`, `answer`; queries `edits`, `inbox` |
| `app/ui/` | the signed publication | dashboard (`app.mjs`, `charts.mjs`, `model.mjs`), training data (`data.mjs`), session connection (`connect.mjs`, `page.mjs`) |
| data registry | read by `data_catalog.py` | every dataset, its producer, inputs, license and consumers: `natlang_neuralese.common.data_registry` (git layer `training/corpora` + the producer journal; `scripts/data_registry.py`) |

Private state (handle, signer, epoch key, descriptor, device store) lives in `~/.local/share/tendriller/natlang-ops-app/private/`
(`NATLANG_OPS_PRIVATE`), never in the repository; the release kit and its verified packages are in `.../natlang-ops-app/kit/`;
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

Publish a UI change (one owner of the signer at a time: stop ops-peer first, restart it after):

    node build.mjs && npx tendril --session $PWD/session.mjs --mode own publish ~/.local/share/tendriller/natlang-ops-app/page --operation ops-release-N

**First connection (owner).** Opening the page connects; there are no buttons (Tendriller's connection component,
`app/ui/connection.mjs` + `connect-view.mjs`). A browser that is already registered connects at once, in any tab. A new browser
creates its own key and a pairing request; the ops participant prepares the exact control change within seconds (host notice),
and the page opens tendriller.dev once, where the owner taps **Approve this device**. With a host that supports `return_to`
(Tendriller PR #73) tendriller.dev returns to the page by itself; until that is deployed, come back to the page after approving.
The participant then commits and the page connects. Enrollment stays open while ops-peer runs. The earlier native owner
auto-enrollment (which trusted author-controlled page code) was removed. The app uses the PR #73 SDK build from
`~/.local/share/tendriller/natlang-ops-app/kit-pr73/`.

Changing `domain.mjs` semantics needs a new definition version and a new application setup (`npm run create/init`), not a publish.
First setup (done 2026-11-20): `npm install` of the kit packages, `npm run create -- https://tendriller.dev OWNER "natlang ops"`,
`npm run init`, build, publish. Browser enrollment: see First connection above (open the page, approve once on tendriller.dev).

## Preview and tests

Layout preview with real data (no session; edits stay in memory): `python preview_server.py` under the ledger, then
`http://127.0.0.1:8765/`. Tests: `pytest tests/ops_dashboard`.
