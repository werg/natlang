# Infrastructure implementation status

Implementation branch: `infrastructure-implementation`. The sequence in [INFRASTRUCTURE_IMPLEMENTATION.md](INFRASTRUCTURE_IMPLEMENTATION.md) remains the design and acceptance plan. These packages were implemented as separate commits so the compatibility and behavioral changes can be reviewed independently.

| Patch | Delivered implementation | Primary evidence |
|---|---|---|
| 01 | Versioned compatibility outcomes, actions, ordered effects, and discrepancy list | `conformance/infrastructure_baseline.json`; `tests/test_infrastructure_baseline.py` |
| 02–03 | Model turn protocol, run-owned settings, logical call/attempt identity, versioned derived seeds and separate world RNG | `natlang/invocation.py`; `tests/test_invocation.py` |
| 04 | Injected crisp executor and exact portable value boundary | `natlang/execution.py`; `tests/test_execution_boundary.py` |
| 05 | Checked in-memory definition graph and value binding independent of filesystem lookup | `natlang/codebase.py`, `natlang/host.py`; `tests/test_checked_sources.py` |
| 06–07 | JSONL trace manifest, actions, evals, effects, typed state changes, offline reconstruction and inspection | `natlang/trace.py`, `scripts/inspect_trace.py`; `tests/test_reduction_trace.py` |
| 08 | Explicit tools-v3 `run_code.engine`, authored engine binding, and historical tools-v2 projection | `natlang/surface.py`, `natlang/surface_projection.py`; `tests/test_engine_selection.py` |
| 09 | Optional retained JS environment with bounded file, byte and argv-process operations | `hosts/retained_js.py`, `hosts/retained_js_worker.js`; `tests/test_retained_js_host.py` |
| 10 | Eval-selectable source/type/run operations and bounded child invocations | `natlang/meta.py`; `tests/test_meta.py` |
| 11 | Scoped SQL engine with bound scalar parameters and transactions | `hosts/sqlite_host.py`; `tests/test_sqlite_host.py` |
| 12 | Four-state stream Fold, waiting/resume, bounded consumption and job completion events | `natlang/streams.py`, `hosts/job_stream.py`; `tests/test_stream_fold.py`, `tests/test_job_stream.py` |
| 13 | Whole-run scenario admission and template-neutral teacher collection/replay linked to program IR | `natlang/scenario.py`, `scripts/collect_scenario_teacher.py`, `scripts/materialize_ir.py`, `scripts/materialize_teacher_trajectory_ir.py`; `tests/test_whole_program_teacher.py` |
| 14 | Optional exact search, measured model help and typed two-format rendering | `hosts/search.py`, `hosts/model_help.py`, `hosts/rendering.py`; `tests/test_optional_libraries.py` |
| 15 | Opt-in bounded parallel finite Map and honest windowed stream composition | `natlang/runtime.py`, `natlang/streams.py`; `tests/test_parallel_map.py` |
| 16 | Lightweight browser/JS source, recorded-decision, seed, combinator and trace subset | `web/natlang_lite.mjs`, `web/inspector.html`; `tests/test_portable_embedding.py` |
| 17 | Local Fold checkpoints and external operation receipt reconciliation | `hosts/recovery.py`; `tests/test_recovery.py` |
| TS host | Native TypeScript interpreter, crisp eval with direct native sharing, model/capability callbacks, live Fold streams, desktop bindings, and package | `ts-host/src/native/`, `ts-host/src/browser/`; `ts-host/test/native-host.test.mjs` |

The existing interpreter tool inventory remains unchanged except for the versioned `engine` argument on `run_code`. Optional host libraries are installed with the wheel but are not imported by the core. The Node host is trusted code, not a sandbox. Trace manifests identify each engine's environment lifetime and authority. Trace readers reconstruct captured natlang state and never replay live effects or arbitrary native memory. The TypeScript host runs its own native reducer. [The native port plan](NATIVE_TYPESCRIPT_PORT.md) records its parity work and limits.

## Verification and empirical gates

- The current TypeScript host suite passes **90 Node tests** and **23 native conformance checks**; the full Python suite passes **298 tests, 1 skipped**. The packed npm artifact contains compiled JavaScript, declarations, prelude and README.
- `uv build --out-dir /tmp/natlang-infrastructure-release-final` succeeded. The wheel contains `hosts/retained_js_worker.js`, `hosts/recovery.py`, `web/natlang_lite.mjs`, `web/inspector.html`, and the core runtime. The retained JS host requires Node.js; the build added no new Python dependencies.
- The paired real-model surface probe is implemented in `scripts/probe_engine_surface.py`. Its recorded-driver test passes. It has not been run against the local Bonsai server because the prior teacher backfill is actively using its single model slot. Model reproducibility and student success are measured properties of a selected backend, not consequences of the seed API alone.
- The TypeScript host runs without Python and directly receives application-owned Node objects. Its Node integration suite covers checked sources, model turns, Map, Fold, declared capabilities, traces, files, processes, retained identity, and failure effects. `plans/NATIVE_TYPESCRIPT_PORT.md` records its design and authority limits.
- The browser embedding supports the declared portable fixture subset. It does not claim general language compatibility, local inference, shell execution, or arbitrary native-state checkpointing.
- Whole-program teacher collection and materialization round trip nested Map and correct blocker fixtures. Large teacher shards and template-rendered student pilots should be collected only after the paired live-model probe and the desired student surface are selected.
- The implementation plan's empirical gates remain open: the paired live-model tools-v2/tools-v3 comparison, selected student pilot, and named P01/P03/P04/P10/P13/P18 application demonstrations. The delivered fixtures show the required mechanisms but do not measure those applications or student behavior. The one-slot Bonsai server remains occupied by the teacher backfill.

## Typical commands

```bash
.venv/bin/python -m pytest -q
uv build --out-dir /tmp/natlang-infrastructure-dist
.venv/bin/python scripts/materialize_ir.py data/programs.ir.jsonl data/turns.jsonl --trace-dir data/traces
.venv/bin/python scripts/collect_scenario_teacher.py data/programs.ir.jsonl data/teacher.jsonl --model-id MODEL --root-seed 43 --limit 1
.venv/bin/python scripts/materialize_teacher_trajectory_ir.py data/teacher.jsonl data/teacher-turns.jsonl
.venv/bin/python scripts/probe_engine_surface.py --model-id MODEL --out runs/engine-surface-pilot.json
```

Use actual existing IR paths and the intended model identity in those commands. The first corpus operation writes trace-linked turns; the teacher command records raw decisions in template-neutral IR; export through `scripts/export_sft.py` selects the current model template later.
