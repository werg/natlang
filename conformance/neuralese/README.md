# Neuralese conformance cases

Cases for the Neuralese extension (spec/SPEC.md "Neuralese", "Contexts", "Captures",
"Iteration and termination", and the `spec/NEURALESE_*.md` documents).

- `types/` holds TypeScript programs checked against `spec/neuralese.d.ts`. They pass
  now:

  ```sh
  node_modules/.bin/tsc -p conformance/neuralese/types/tsconfig.json
  ```

  `valid.ts` must compile; every line in `errors.ts` under `@ts-expect-error` must
  fail.
- `cases/` holds behavioural cases. Every case has `status: pending` until the stage
  named in `stage` implements it (S3 port, S4 runtime and servers, S5 program
  training). An implemented case names its runnable form in `implemented_by`. The
  conformance runner (`ts-host/scripts/native-conformance.mjs`, `npm run
  test:conformance`, the `ts-conformance` step of `scripts/check-main.sh`) checks that
  every form exists and runs the forms under `ts-host/test/conformance/` itself; forms
  that are test files run with their suite. A case whose `engines` lists
  `python-native` runs against a real Python native engine on a tiny backbone
  (`tests/neuralese/learning_conformance_server.py`; Python from `NATLANG_PYTHON`, else
  `.venv-neuralese`). `NATLANG_CONFORMANCE_CASES=id,id` runs only those cases.

## Case fields

| Field | Meaning |
| --- | --- |
| `id`, `title` | Stable identifier and summary. |
| `stage` | Stage that makes it pass: `S3`, `S4`, or `S5`. |
| `status` | `pending`, or `implemented` with `implemented_by`. |
| `engines` | Engines an implemented case runs on (`python-native`); absent: no model server. |
| `spec` | Section(s) the case checks. |
| `given` | Program, files, values, or server state. |
| `expect` | Diagnostic code, value, error, trace property, or agreement. |
