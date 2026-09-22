# Hosts and model adapters

## Pick the actual host

| Target | Public path | Crisp execution |
|---|---|---|
| Python compatibility | `natlang.host.load` / `load_definitions`, `Runtime`, `ToolAgent` | Historical traces and offline utilities; not the canonical runtime or teacher collector |
| Node | `NatlangHost` from `@natlang/typescript-host` | Canonical TypeScript host for crisp code and model eval; Python is not a production dependency |
| Browser | `BrowserNatlangClient`, `BrowserNatlangApplication` from `@natlang/browser` | Browser TS evaluator and local model or supplied driver |

The inspected TS package is private and built from the repository. Do not promise a public npm package or downloadable model without checking availability. In a checkout, `npm ci --prefix ts-host` then `npm --prefix ts-host run build` builds it. A consumer can install the built local directory. Python can be installed with `pip install -e '.[js,dev]'` from the checkout; the optional JS dependency is needed for default crisp execution. Source programs themselves are regular files or virtual file maps.

## Python

```python
from pathlib import Path
from natlang.host import load
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.invocation import RunOptions, SeedPolicy

# decoder is supplied by the embedding and implements natlang's chat contract.
def classify(decoder, text):
    root = load(Path("inspect.nl"), {
        "text": text,
    })
    runtime = Runtime(
        lambda lam: ToolAgent(decoder, validation_feedback="local"),
        options=RunOptions(seed=SeedPolicy("derived", 17)),
    )
    outcome, value = runtime.run_root(root)
    return outcome, value
```

Records passed as ordinary inputs remain ordinary typed data and can be read or
passed to children through normal TypeScript expressions. Filesystem access is
separate: only directory reducers receive model-facing file tools and the
relative-path folder API.

Inspect `outcome.kind` before using the value as completed. `load()` can interpret existing short string paths as files; use `load_definitions(entries, root_name, inputs)` for explicit in-memory values when that ambiguity matters. Checked definition entries use `args`, `returns`, exactly one of `instructions` or `code`, and optional `types`, `uses`, `effects`, `engine`.

`Runtime` supports declared capabilities and a registry of crisp executors. Consult `natlang/execution.py` for `CrispRequest`, the executor protocol, and portable value validation. An evaluator must report errors and return exact portable values; do not let arbitrary native objects silently enter the typed tree. Engine metadata should accurately describe environment lifetime, authority, and replay limits.

## Node

```ts
import { NatlangHost } from '@natlang/typescript-host';

export async function classify(modelTurn, text) {
  const host = new NatlangHost();
  try {
    const result = await host.run({
      source: { kind: 'file', path: 'inspect.nl' },
      inputs: { text },
      modelTurn,
      validationFeedback: 'local',
      options: { seed: { mode: 'derived', root: 17 } },
    });
    if (result.outcome.kind !== 'done') throw new Error(result.outcome.detail);
    return result; // value, outcome, trace, run identity, and emitted observations
  } finally {
    host.close();
  }
}
```

Node also accepts `source: {kind:'definitions', entries, root}` and
`source: {kind:'program', program}`. Browser source files use a separate
`kind:'files'` contract. That source map defines natlang source; it is not
automatically a semantic argument. Pass application data as ordinary typed
values such as `Record<string, T>`.

For reads or writes against an input folder, call a directory reducer. Its
paths are relative to that folder, and `folder.apply` is the explicit operation
that retains selected edits. Use crisp host functions for filesystem access
outside the reducer's supplied folder.

Supply `host: applicationObjects` and `mode:'retained'` when their identity/lifetime is needed. If constructing and supplying a `TypeScriptEnvironment` yourself, retain ownership and close it yourself. Close native bindings/jobs separately according to their API; closing an interpreter is not guaranteed to terminate host-owned work.

## Model transport

The TS `modelTurn` callback receives `messages`, `tools`, `temperature`, `seed`, and `max_tokens` and returns:

```ts
{
  calls: [['eval', { code: 'const result = await helper(sample, "label"); result' }]],
  text: '',
  completion_tokens: 42,
  prompt_tokens: 700,
}
```

The example illustrates the transport shape, not a canned interpreter policy. Return actual model output. Empty calls end a normal interpreter episode; checkpoint requests intentionally have no work tools and ask for a note. Preserve tool-call identities and every per-call result. A response may contain an ordered batch. Do not discard all but the first call or require serial generation just because the runtime applies effects in order.

Keep model/template quirks inside the adapter. Source programs use `eval`,
`mark_lines`, and ordinary positional calls; adapters may map those tools to a
provider's structured representation. Use the selected backend's supported
typed tool representation. Do not silently unwrap malformed values or paste
model-specific conventions into `.nl` files. Capture the exact presented
tools/messages, not only the pre-adaptation request.

The shared OpenAI-compatible adapter retries one response whose tool arguments
are malformed JSON, adding a narrow correction to the original turn. A second
malformed response is a bounded transport failure. This repair happens before
runtime validation feedback, which handles well-formed calls that violate the
offered schema or current state.

When `max_tokens` is absent/null, omit a provider field that requires an integer. Report actual usage and termination reasons when available. Do not replace unknown usage with invented zero-cost success. Forward cancellation where supported; provider cancellation and effect cancellation are different.

## Version checks

Use `natlang/runtime.py`, `natlang/tool_agent.py`, `natlang/host.py`, `ts-host/src/contracts.ts`, `ts-host/src/native/host.ts`, and `ts-host/src/browser/host.ts` as API anchors. The older `web/natlang_lite.mjs` helpers are useful trace/experimental utilities, not a substitute for the complete native browser interpreter.
