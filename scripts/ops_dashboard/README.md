# ops dashboard exporter and page

`export.py` feeds the "natlang ops" live dashboard (a Tendriller page, contract 3). Every 30 s it writes `latest.json` (the whole current state as
one record, replaced atomically) and appends machine history, metric and event JSONL to `<data_dir>/ops/dashboard/out/` (`config.json`: `data_dir`,
floors, Pop ssh host, `runs_window_days`, `machine_history_s`, `teacher_pool`). The Tendriller collector (`tendril-stream.mjs` from the core tools
release, scoped producer credential) uploads the files to streams. Data contract and event receiver: `<data_dir>/ops/dashboard/CONTRACT.md`.

`state.json` beside the output keeps byte offsets of each `train.jsonl`, units seen, held results and the teacher history, so a restart neither
repeats nor loses records. Pop is read with one `ssh POP python3 - --host-json CFG < export.py` call per cycle (the same file runs there).
Annotations (titles, queue, investigations) are re-read from `../annotations.json` each cycle.

Run (ledger units, low priority, ~0.3 GB each; use a fresh numeric suffix, journals of old unit names persist):

    memory_ledger.py run --unit ops-dashboard-export-N --budget-gb 0.3 --class service --priority low --wait 120 --detach -- /usr/bin/python3 ~/natlang/scripts/ops_dashboard/export.py
    memory_ledger.py run --unit ops-dashboard-collector-N --budget-gb 0.3 --class service --priority low --wait 120 --detach -- NODE ~/.local/share/tendriller/natlang-ops-v3/tools/bin/tendril-stream.mjs ~/.local/share/tendriller/natlang-ops-v3/collector.json

Collector provisioning (once per page, from the creation response saved as `create.json`): write `streams.json` (streams and sources together) and
`tendril-stream.mjs init --page create.json --streams streams.json --out collector.json`.

One-shot: `python3 export.py --once --out DIR` (options `--ops --annotations --interval --no-pop`). Test: `pytest tests/ops_dashboard`.

## Page (`page/`)
Plain ES modules (`app.mjs` views and intake, `charts.mjs` SVG charts, `model.mjs` pure helpers), no vendored components. Preview against real exporter output
(needs Playwright resolvable from the working directory and a matching Chromium):

    python3 page/make_fixture.py DIR > fixture.json
    node tendril-preview.mjs scripts/ops_dashboard/page --fixture fixture.json     # loopback URL

Publish: `scripts/ops_dashboard/publish.sh "message"` (stages only the page files and runs `tendril push`; handle `natlang-ops-v3/create.json`).
Wait for the page's events: `scripts/ops_dashboard/wait_event.sh` (see CONTRACT.md).
