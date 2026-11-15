# ops dashboard exporter

`export.py` feeds the "natlang ops" live dashboard. Every 30 s it appends whole JSONL lines (collector envelope
`{"ts", "type", "data"}`, each < 30 KiB) to four files in
`/home/werg/data/lfm3-rl-dgx-20261029/ops/dashboard/out/`:

| file | content |
|---|---|
| `machines.jsonl` | DGX and Pop: CPU, load, memory, swap, GPU, memory-ledger summary, disks with floors (every 60 s per machine) |
| `metrics.jsonl` | `train`: one record per learner step of every `ops/**/train.jsonl` (depth <= 3, flattened to ~80 numeric fields); `held`: one per `held/*/compare-vs-step0.json` when it appears or changes |
| `snapshot.jsonl` | `processes` (every cycle; ledger claims + systemd units, finished/failed kept 6 h), `queue`, `investigations`, `alerts`, `fallow` (rows of `plans/neuralese/FALLOW_REGISTER.md`); the slow types only on change or every 10 min; big lists split over records with `data.part`/`data.parts` |
| `events.jsonl` | `event`: unit started / finished / failed, held results, grad_spike / reasoning_shift / behaviour_shift steps |

`state.json` in the same directory keeps byte offsets of each train.jsonl, units seen so far and emission signatures,
so a restart neither repeats nor loses records. Pop is read with one `ssh pop-os python3 - --host-json CFG < export.py`
call per cycle (the same file runs there); an unreachable Pop becomes an alert. Annotations (titles, queue,
investigations) are re-read from `../annotations.json` each cycle.

Run: `python3 export.py` (loop), `python3 export.py --once` (one cycle), options `--ops --out --annotations --interval
--no-pop`. Start it as a ledger unit, e.g. `memory_ledger.py run --unit ops-dashboard-export --budget-gb 2 -- /usr/bin/python3
/home/werg/natlang/scripts/ops_dashboard/export.py`. Test: `pytest tests/ops_dashboard`.

Data contract: `/home/werg/data/lfm3-rl-dgx-20261029/ops/dashboard/CONTRACT.md`.
