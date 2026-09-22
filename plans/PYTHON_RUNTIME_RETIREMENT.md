# Python runtime retirement inventory

Status: migration in progress. New whole-program `scope-eval-v1` teacher runs
start at `node ts-host/scripts/teacher-collector.mjs`. That stable command
delegates to the compiled native collector at `ts-host/dist/teacher/cli.js` and
fails closed instead of falling back to Python.

This inventory covers executable paths that import `natlang.runtime.Runtime` or
`Session`, and commands that start those paths. Python data transforms that do
not execute a natlang program can move later without blocking runtime retirement.

## Active teacher generation

| Role | Current files | Retirement state |
| --- | --- | --- |
| Coverage pipeline | `scripts/run_teacher_generation.sh` | Starts the stable Node collector command. |
| Playground teacher job | `ts-host/scripts/playground-jobs.mjs` | Starts the stable Node collector command with a durable jobs directory. |
| Whole-program collectors | `scripts/collect_scenario_teacher.py`, `scripts/collect_teacher_batch.py` | Kept for imports and historical tests; their CLIs refuse new `scope-eval-v1` runs. |
| Generative leaf collector | `scripts/teacher_leaves.py` | Still executes Python `Runtime`; port its reference checks and audit capture. |
| Specialized collector | `scripts/collect_semantic_merge_teacher.py` | Still executes Python `Runtime`; replace with Node scenarios. |
| Teacher probes | `scripts/teacher_behavior_probe.py`, `scripts/confidence_probe.py`, `scripts/marking_probe.py`, `scripts/validation_probe.py`, `scripts/application_probe.py`, `scripts/probe_engine_surface.py`, `scripts/compare_continuations.py`, `scripts/baseline.py`, `scripts/paraphrase.py` | Still execute or construct Python runtime sessions; port only probes still used as release evidence. |
| Backfill launcher | `scripts/finalize_synthetic_backfill.py` | Starts `teacher_leaves.py` for missing references. |

Teacher documentation and commands that still describe a Python collector are
in `TEACHER_SETUP.md`, `TEACHER_TRAJECTORY_IR.md`, `README.md`,
`plans/CONTINUATION_CHECKPOINTS.md`, and
`plans/INFRASTRUCTURE_IMPLEMENTATION_STATUS.md`. Historical results may keep
their original command names, but runnable examples must use the Node command.

## Production and application execution

These Python entrypoints still instantiate `Runtime` or `Session` and therefore
remain migration work:

- Installed Python command and embedding core: `natlang/__main__.py`,
  `natlang/host.py`, and `natlang/meta.py`. These construct the implementation
  in `natlang/runtime.py`; remove them or turn them into explicit Node process
  adapters before deleting that module.
- Applications: `applications/data_migration.py`,
  `applications/experiment_lab.py`, `applications/test_explorer.py`, and
  `applications/type_studio.py`.
- Application and service commands: `scripts/run_data_migration.py`,
  `scripts/run_experiment_lab.py`, `scripts/run_test_explorer.py`,
  `scripts/run_type_studio.py`, `scripts/run_application_evals.py`,
  `scripts/serve_web.py`, and the Python `natlang run` command documented in
  `README.md` and `spec/SPEC.md`.
- Corpus/reference execution: `scripts/build_synthetic_ir.py`,
  `scripts/generate.py`, `scripts/generate_agent_support.py`,
  `scripts/generate_failures.py`, `scripts/materialize_teacher_trajectory_ir.py`,
  and `scripts/program_ir.py`.

`scripts/program_ir.py` imports `OpenList` lazily. It still couples offline IR
work to Python runtime values even where no model execution occurs.

Runnable Python execution examples remain in `README.md`, `AGENT_SUPPORT.md`,
`CAREFUL_MODE.md`, `TEACHER_SETUP.md`, `spec/SPEC.md`,
`codebases/semantic_merge/README.md`, and
`plans/INFRASTRUCTURE_IMPLEMENTATION_STATUS.md`. Port each command when its
corresponding Node application, probe, or collector is ready; do not relabel
historical run evidence as Node output.

## Tests coupled to Python execution

The following test modules directly import `Runtime` or `Session` and must be
ported to Node coverage or retired with the Python runtime:

`test_agent_support.py`, `test_careful.py`, `test_checked_sources.py`,
`test_codebase.py`, `test_conformance_reference.py`, `test_continuation.py`,
`test_directory_reducer.py`, `test_distributed_skills.py`,
`test_engine_selection.py`, `test_execution_boundary.py`,
`test_explicit_surface.py`, `test_highlighter.py`,
`test_infrastructure_baseline.py`, `test_invocation.py`, `test_job_stream.py`,
`test_legal_move.py`, `test_marking_probe.py`, `test_meta.py`,
`test_moderation.py`, `test_nlprolog.py`, `test_parallel_map.py`,
`test_portable_embedding.py`, `test_recovery.py`, `test_reduction_trace.py`,
`test_retained_js_host.py`, `test_review_fixes.py`, `test_scenario_admission.py`,
`test_scope_eval.py`, `test_semantic_merge_app.py`,
`test_semantic_merge_menagerie.py`, `test_shopkeeper.py`,
`test_spell_arena_app.py`, `test_stream_fold.py`,
`test_structured_session.py`, `test_surface.py`, `test_teacher_adapter.py`,
`test_teacher_behavior_probe.py`, `test_validation_feedback.py`, and
`test_webserver.py`.

Teacher collector tests also import the retained Python implementation:
`test_collect_teacher_batch.py`, `test_continuation.py`,
`test_teacher_trajectory_ir.py`, and `test_whole_program_teacher.py`. Keep these
as legacy characterization tests until the Node collector has equivalent
resumption, provenance, replay, and admission tests.

## Deletion gate

Delete no Python runtime or collector files until the Node collector passes
focused whole-program cases, interruption/resume, trace admission, and output
provenance checks. Then replace legacy characterization tests with Node tests,
remove the guarded Python CLIs, and rescan this inventory with:

```bash
rg -l 'from natlang.runtime import (Runtime|.*Session)|import natlang.runtime' \
  scripts applications hosts web tests
```
