# Native conformance programs

`programs/` contains small typed programs that exercise the interpreter and runtime. Each
fixture is a named natural-language function with its callable folder, inputs, and the expected
output.

## Fixture fields

- `root` and `files`: a small project. The root is a named `.nl` function; its companion folder
  holds the natural-language and TypeScript functions it may call, exactly as on disk.
- `inputs`: values bound by parameter name.
- `expect.value`: exact final typed value.
- `expect.checks`: optional `crisp` assertions over `value`, or `judge` questions for non-unique prose.
- `expect.status` and `note_checks`: expected blocked outcome (`quiesced`) and its reason.
- `expect.emitted`: records sent to the `out` service (`import { out } from 'natlang:services'`).
- `lint.must` and `lint.must_not`: intended trace properties for teacher-led evaluation.
- `reference`: a scripted path per function name, run in place of a model.

Reference entries use `eval` with ordinary TypeScript statements; the final expression is the
function's result. Called functions take positional arguments. A function may instead use
`answer`, `answer_by`, or `blocker`, supplied through the same eval or blocker tool the model has.
The driver closes the instruction lines with `mark_lines` before ending the invocation.

Fixtures use ordinary TypeScript types and control flow. Callable-folder TypeScript follows the
finite-iteration policy: bounded loops, `for...of`, and array methods, with no recursion.

## Running

```sh
npm run test:conformance --workspace @natlang/typescript-host
```

`native-conformance.mjs` runs each reference through the built runtime with the reference agent in
place of a model. A fixture whose reference is marked `known_issue` is skipped.
