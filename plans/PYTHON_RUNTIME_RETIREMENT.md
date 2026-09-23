# Python runtime retirement (complete)

The Python runtime (`natlang/`), its host bindings (`hosts/`), the lite browser
interpreter (`web/`), the Python applications (data migration, experiment lab,
test explorer, type studio), and every script and test that executed the Python
runtime were removed in the TypeScript-native refactor. Their roles are now
covered as follows:

| Former Python role | Current implementation |
|---|---|
| Runtime, surfaces, eval, prompts | `ts-host/src/native/`, `ts-host/src/runtime/` |
| Whole-program teacher collection | `ts-host/scripts/teacher-collector.mjs` (`ts-host/src/teacher/collector.ts`) |
| Program IR builders and synthetic generation | `ts-host/src/teacher/program.ts`, `ts-host/scripts/generate-synthetic-ir.mjs`, `freeze-source-teacher-cases.mjs`, `build-read-before-code-probe.mjs` |
| Trajectory materialization and SFT export | `ts-host/src/teacher/native-materializer.ts`, `ts-host/scripts/export-native-sft.mjs` |
| Playground case import | `ts-host/scripts/import-playground-cases.mjs` |
| Conformance checks | `ts-host/scripts/native-conformance.mjs` over `conformance/programs/` |
| Python applications | Studio apps P05, P13, P16, P17 (`ts-host/studio/`); the named programs remain in `codebases/` as corpus |

Program IR moved to `natlang.program/2`: a root `.nl` file and its callable
folder instead of a `$lambda` tree. `ts-host/scripts/migrate-program-ir.mjs`
upgraded the tracked IR files once. Historical run evidence, trajectories, and
archives keep their original formats and command names.

The remaining Python code is the runtime-free training toolchain: corpus
selection and rendering, curriculum preparation, and LoRA training
(`scripts/*.py`, `tests/`, `pyproject.toml` as `natlang-training`).
