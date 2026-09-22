# Native conformance programs

`programs/` contains small typed programs that exercise the TypeScript scope runtime.
Each fixture declares a root function, optional imported natural language and TypeScript
functions, inputs, and expected output.

## Fixture fields

- `program`: the serialized test function and its inline codebase.
- `program_file`: an authored natural language function loaded from a codebase file.
- `inputs`: values bound by parameter name for the test driver.
- `expect.value`: exact final typed value.
- `expect.checks`: optional `crisp` assertions or `judge` questions for non-unique prose.
- `expect.status` and `note_checks`: expected blocked outcome and its reason.
- `expect.emitted`: expected effect records.
- `lint.must` and `lint.must_not`: intended trace properties for teacher-led evaluation.
- `reference`: a scripted native-runtime path for deterministic regression coverage.

Reference entries use `eval` with ordinary TypeScript expressions and statements. Imported
functions are called positionally, and the final expression supplies the function result. A
leaf may instead use `answer`, `answer_by`, or `blocker`; these values are supplied through
the same eval or blocker tool exposed to the runtime. The driver closes the instruction lines
with `mark_lines` before ending the episode.

Fixtures use ordinary TypeScript types and control flow. List processing, loops, and
accumulator updates are written with JavaScript and TypeScript constructs such as `map`,
`for...of`, and `Record<string, T>`.

## Running

```sh
npm run test:conformance --workspace @natlang/typescript-host
```

`native-conformance.mjs` runs each supported reference against the built native runtime. A
fixture marked `known_issue` remains visible in the corpus but is skipped until it is rewritten
for the current surface.
