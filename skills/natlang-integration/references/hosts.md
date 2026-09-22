# Hosts and model adapters

## Pick the host

| Target | Public API | Model execution |
|---|---|---|
| Node | `NatlangHost` from `@natlang/node` | Supplied `modelTurn` or a configured local model driver |
| Browser | `BrowserNatlangClient` and `BrowserNatlangApplication` from `@natlang/browser` | Local browser model or supplied `modelTurn` |

The packages are built from this repository. See [native packages and
executables](../../../NATIVE_PACKAGES.md) for installation and package details.
For checkout development, run `scripts/setup_dev.sh --node-only` and
`npm --prefix ts-host run build`.

## Node

```ts
import { NatlangHost } from '@natlang/node';

const host = new NatlangHost();
try {
  const result = await host.run({
    source: { kind: 'file', path: 'inspect.nl' },
    inputs: { text: 'The trial improved response times.' },
    modelTurn,
  });
  if (result.outcome.kind !== 'done') throw new Error(result.outcome.detail);
  return result.value;
} finally {
  host.close();
}
```

Node also accepts checked definition graphs and program values. Provide
application data as ordinary typed values, including `Record<string, T>`.
For filesystem work, call a directory reducer with an explicit `Folder`
argument. The supplied folder is the reducer's writable root; paths are
relative to it. A direct call `await reducer(folder, ...args)` returns its
typed value and discards its edits. `await folder.apply(reducer, ...args)`
retains the reducer's committed changes. A selected subdirectory can be passed
with `folder.dir('subdirectory')`.

## Browser

The browser source map defines natlang source files; it is not automatically a
semantic input. Pass application data through `inputs`. Browser source files
may include ordinary default-exported TypeScript helpers and natural-language
`.nl` functions. Reducer folder handles are provided by the application when a
directory reducer is called.

Use `BrowserNatlangClient` for model loading and one-run execution. Use
`BrowserNatlangApplication` when state updates, rendering, and event ordering
belong to a UI lifecycle. Supply `modelTurn` to connect a different backend or
a scripted fixture. Label fixtures distinctly from live-model results.

## Model transport

The TypeScript `modelTurn` callback receives `messages`, `tools`, `temperature`,
`seed`, and `max_tokens`. It returns ordered tool calls and optional text and
usage fields:

```ts
{
  calls: [['eval', { code: 'const result = await helper(sample); result' }]],
  text: '',
  completion_tokens: 42,
  prompt_tokens: 700,
}
```

Return the provider's actual output and preserve call order, IDs, and tool
results. Keep provider-specific formatting in the model adapter. The program
surface remains TypeScript `eval`, scope inspection, line marking, and normal
function calls.

If `max_tokens` is absent or null, omit a provider field that requires an
integer. Record actual usage and finish reasons where available. Forward
cancellation when supported; provider cancellation and host-effect
cancellation are separate operations.

## Host authority

Native APIs, databases, processes, and other host objects remain owned by the
embedding. Expose them through ordinary crisp helpers or declared effect
callbacks. The shared TypeScript evaluator is trusted application code, not a
sandbox. A timeout or failed result cannot undo a host mutation that already
occurred. Persist operation identities and observations when a retry could
duplicate an external effect.
