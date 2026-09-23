# TypeScript runtime for natlang

The Node and browser runtimes share a TypeScript implementation of source
loading, typed function calls, model tools, and directory reducer behavior.
Node applications import `NatlangHost` from `@natlang/node`; browser
applications use `@natlang/browser`. See [native packages and
executables](../NATIVE_PACKAGES.md) and [development setup](../DEV_SETUP.md).

## Build

```sh
cd ts-host
npm ci
npm run build
npm test
```

Node 22.13 or newer is required. The runtime uses TypeScript's compiler API to
transpile crisp modules and eval snippets. Eval snippets are checked against
the shared runtime value model at transaction and function boundaries; the
runtime does not promise full TypeScript type checking for model-generated
snippets.

## Source files

A crisp module is ordinary TypeScript with one default-exported function:

```ts
export default function countWords(text: string): number {
  return text.trim().split(/\\s+/).filter(Boolean).length;
}
```

Natural-language functions use typed frontmatter and instruction lines in a
`.nl` file. Both kinds of function use standard TypeScript types, positional
parameters, and imports. Call synchronous TypeScript functions directly and
await asynchronous TypeScript or natural-language functions. Refer to [the specification](../spec/SPEC.md)
for source, type, and completion rules.

## Run

```ts
import { NatlangHost } from '@natlang/node';

const host = new NatlangHost();
try {
  const result = await host.run({
    source: { kind: 'file', path: 'inspect.nl' },
    inputs: { text: 'The trial improved response times.' },
    modelTurn,
  });
  console.log(result.outcome, result.value);
} finally {
  host.close();
}
```

The model receives `eval`, `read_value`, and `mark_lines`, plus blocker/error
reporting and function inspection/editing tools. Parameters, imported
functions, and persistent locals are lexical names in eval. Imported natural
language and TypeScript functions are called normally.

## Codebase editing and directory reducers

Models are encouraged to inspect and improve existing imported instructions
and TypeScript helpers through the function editing tools. The set of
codebase functions stays fixed during a run: those tools edit existing source
but do not add, delete, move, or rename functions.

Only directory reducers receive model-facing filesystem tools. The reducer's
first parameter is an explicit `Folder`; inside the reducer it is available as
`folder`. Paths are relative to that folder, with no root prefix. A direct
`await reducer(folder, ...args)` returns its typed value and discards edits.
`await folder.apply(reducer, ...args)` retains committed edits. To delegate a
subdirectory, use `await folder.dir('subdirectory').apply(reducer, ...args)`.
The child receives that subdirectory as its own `folder` root. Reducers may
also be called directly when only their typed return is needed.

The detailed file API and reducer call behavior are in the reducer-specific
prompt at `natlang/prompts/tools_directory_reducer.md`. Normal lambdas do not
receive that API.

## Browser

Use [`BROWSER_CLIENT.md`](BROWSER_CLIENT.md) for local model loading, WebGPU,
asset hosting, and browser lifecycle. Use
[`TERMINAL_APPLICATIONS.md`](TERMINAL_APPLICATIONS.md) for Node CLI and
terminal application patterns.

## Trust and host effects

The TypeScript evaluator executes trusted application code and is not a
sandbox. Host objects passed into the environment remain accessible by
identity. A failed eval, validation error, or cancellation cannot undo a native
operation that already occurred. Keep effect identity and observations in the
application when retrying external operations.
