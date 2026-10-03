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
  training). The native conformance runner (`ts-host/scripts/native-conformance.mjs`)
  reads `conformance/programs/` only, so pending cases do not run. When a case is
  implemented, it moves to a runnable form (a program fixture, a test, or the server
  parity suite) and its entry records where.

## Case fields

| Field | Meaning |
| --- | --- |
| `id`, `title` | Stable identifier and summary. |
| `stage` | Stage that makes it pass: `S3`, `S4`, or `S5`. |
| `status` | `pending`, or `implemented` with `implemented_by`. |
| `spec` | Section(s) the case checks. |
| `given` | Program, files, values, or server state. |
| `expect` | Diagnostic code, value, error, trace property, or agreement. |
