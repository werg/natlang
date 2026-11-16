# ops dashboard exporter and page

`export.py` feeds the "natlang ops" live dashboard (a Tendriller page). Every 30 s it writes `latest.json` (the whole current state as one
record, replaced atomically) and appends machine history, metric and event JSONL to `<data_dir>/ops/dashboard/out/` (`config.json`:
`data_dir`, floors, Pop ssh host, `runs_window_days`, `machine_history_s`, `teacher_pool`). A Tendriller collector (`tendril-stream.mjs`,
scoped producer credential) uploads them to streams. Data contract: `<data_dir>/ops/dashboard/CONTRACT.md`.

`state.json` beside the output keeps byte offsets of each `train.jsonl`, units seen, held results and the teacher history, so a restart
neither repeats nor loses records. Pop is read with one `ssh POP python3 - --host-json CFG < export.py` call per cycle (the same file runs
there). Annotations (titles, queue, investigations) are re-read from `../annotations.json` each cycle.

Run (ledger units, low priority, ~0.5 GB each):

    memory_ledger.py run --unit ops-dashboard-export-N --budget-gb 0.5 --class service --priority low --wait 600 --detach -- /usr/bin/python3 ~/natlang/scripts/ops_dashboard/export.py
    memory_ledger.py run --unit ops-dashboard-collector-N --budget-gb 0.5 --class service --priority low --wait 600 --detach -- NODE ~/.local/share/tendriller/natlang-ops-v2/bin/tendril-stream.mjs ~/.local/share/tendriller/natlang-ops-v2/collector.json

One-shot: `python3 export.py --once --out DIR` (options `--ops --annotations --interval --no-pop`). Test: `pytest tests/ops_dashboard`.

## Page (`page/`)
Plain ES modules plus two vendored contrib components (`assets/components/chart.js`, `stream-view.js`, locked in `tendril.assets.json`).
Edit, then preview against real exporter output:

    python3 page/make_fixture.py DIR > fixture.json
    node tendril-preview.mjs scripts/ops_dashboard/page --fixture fixture.json     # loopback URL

Publish from a copy without `make_fixture.py` (the CLI uploads every file of the directory):
`mkdir stage && cp -r page/{index.html,style.css,app.js,model.mjs,tendril.json,tendril.assets.json,assets} stage/ &&
TENDRIL_API=... TENDRIL_TOKEN=... node tendril.mjs push stage` (the handle is `~/.local/share/tendriller/natlang-ops-v2/create.json`).
