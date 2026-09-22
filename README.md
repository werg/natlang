# natlang

Natlang is a TypeScript runtime for programs that combine ordinary TypeScript
functions with natural-language functions interpreted line by line by a model.
Both kinds of function use the same TypeScript types, positional parameters,
imports, and `await`-based call syntax.

## Get started

Set up a checkout and build the Node host:

```sh
scripts/setup_dev.sh --node-only
cd ts-host && npm test
```

The development command accepts a source file or codebase directory:

```sh
natlang --doctor --json
natlang codebases/semantic_terminal
natlang summarize the functions in this codebase
```

See [Development setup](DEV_SETUP.md) for model setup, CLI applications,
browser builds, and runtime commands.

## Source model

Natural-language functions are `.nl` files with typed YAML frontmatter and
instruction lines. Crisp helpers are ordinary `.ts` modules with one default
exported function. For example:

```ts
export default function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}
```

The interpreter uses `eval(code)` for TypeScript, `read_value` to inspect its
scope, and `mark_lines` to close instruction lines after successful work.
`report_blocker` and `report_error` provide explicit failure exits. Imported
functions are available by name and are meant to be inspected and improved
when that would clarify or correct the program. Function editing changes
existing source; it does not add, move, or remove functions.

Directory reducers have a separate prompt and receive a writable folder API.
Their first parameter is an explicit `Folder` value. A direct call such as
`await reduce(folder, input)` returns the typed result and discards file edits;
`await folder.apply(reduce, input)` retains the reducer's committed changes.
Subfolders can be selected with `folder.dir("path")`. Paths are relative to
the supplied folder. Ordinary functions do not receive filesystem tools.

Read the [language specification](spec/SPEC.md), [type guide](TYPES.md), and
[natlang skills](skills/README.md) for the complete contracts.

## Project guides

| Guide | Contents |
|---|---|
| [Native packages](NATIVE_PACKAGES.md) | Node, browser, and core packages |
| [Training](TRAINING.md) | Teacher data, student training, and evaluation |
| [Teacher setup](TEACHER_SETUP.md) | Teacher runtime and collection workflow |
| [Program IR pipeline](PROGRAM_IR_PIPELINE.md) | Portable trajectory representation and materialization |
| [Code corpus](plans/CODE_CORPUS.md) | Application source corpus and processing |
| [Terminal applications](ts-host/TERMINAL_APPLICATIONS.md) | CLI and terminal application framework |

`PLAN.md` and files under `plans/` contain research history and proposals. Use
`spec/SPEC.md` and the current TypeScript source as the active runtime contract.
