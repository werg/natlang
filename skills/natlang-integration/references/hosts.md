# Hosts and model adapters

## Pick the actual host

| Target | Public path | Crisp execution |
|---|---|---|
| Python | `natlang.host.load` / `load_definitions`, `Runtime`, `ToolAgent` | Default isolated QuickJS; custom registered executors |
| Node | `NatlangHost` from `@natlang/typescript-host` | Native TS host; Python is not a production dependency |
| Browser | `BrowserNatlangClient`, `BrowserNatlangApplication` from `@natlang/typescript-host/browser` | Browser TS evaluator and local model or supplied driver |

The inspected TS package is private and built from the repository. Do not promise a public npm package or downloadable model without checking availability. In a checkout, `npm ci --prefix ts-host` then `npm --prefix ts-host run build` builds it. A consumer can install the built local directory. Python can be installed with `pip install -e '.[js,dev]'` from the checkout; the optional JS dependency is needed for default crisp execution. Source programs themselves are regular files or virtual file maps.

## Python

```python
from pathlib import Path
from natlang.host import load
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.invocation import RunOptions, SeedPolicy

# decoder is supplied by the embedding and implements natlang's chat contract.
def run_review(decoder, observations, criterion):
    root = load(Path("review/review.nl"), {
        "observations": observations, "criterion": criterion,
    })
    runtime = Runtime(
        lambda lam: ToolAgent(decoder, validation_feedback="local"),
        options=RunOptions(seed=SeedPolicy("derived", 17)),
    )
    outcome, value = runtime.run_root(root)
    return outcome, value
```

Inspect `outcome.kind` before using the value as completed. `load()` can interpret existing short string paths as files; use `load_definitions(entries, root_name, inputs)` for explicit in-memory values when that ambiguity matters. Checked definition entries use `args`, `returns`, exactly one of `instructions` or `code`, and optional `types`, `uses`, `effects`, `engine`.

`Runtime` supports declared capabilities and a registry of crisp executors. Consult `natlang/execution.py` for `CrispRequest`, the executor protocol, and portable value validation. An evaluator must report errors and return exact portable values; do not let arbitrary native objects silently enter the typed tree. Engine metadata should accurately describe environment lifetime, authority, and replay limits.

## Node

```ts
import { NatlangHost } from '@natlang/typescript-host';

export async function runReview(modelTurn, observations, criterion) {
  const host = new NatlangHost();
  try {
    const result = await host.run({
      source: { kind: 'file', path: 'review/review.nl' },
      inputs: { observations, criterion },
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

Node also accepts `source: {kind:'definitions', entries, root}` and `source: {kind:'program', program}`. Browser virtual files use a separate `kind:'files'` contract. Do not assume the Node file loader is present in a browser.

Supply `host: applicationObjects` and `mode:'retained'` when their identity/lifetime is needed. If constructing and supplying a `TypeScriptEnvironment` yourself, retain ownership and close it yourself. Close native bindings/jobs separately according to their API; closing an interpreter is not guaranteed to terminate host-owned work.

## Model transport

The TS `modelTurn` callback receives `messages`, `tools`, `temperature`, `seed`, and `max_tokens` and returns:

```ts
{
  calls: [['call', { function: 'helper', to: 'let/result', values: { label: 'sample' } }]],
  text: '',
  completion_tokens: 42,
  prompt_tokens: 700,
}
```

The example illustrates the transport shape, not a canned interpreter policy. Return actual model output. Empty calls end a normal interpreter episode; checkpoint requests intentionally have no work tools and ask for a note. Preserve tool-call identities and every per-call result. A response may contain an ordered batch. Do not discard all but the first call or require serial generation just because the runtime applies effects in order.

Keep model/template quirks inside the adapter. For example, the repository's Bonsai adapter aliases `call` because of its parser, while source programs still use the same natlang call semantics. Use the selected backend's supported structured/typed tool representation, including literal parameter types. Do not blindly strip alternatives, silently unwrap malformed values, or paste model-specific conventions into every `.nl` file. Capture the exact presented tools/messages, not only the pre-adaptation request.

When `max_tokens` is absent/null, omit a provider field that requires an integer. Report actual usage and termination reasons when available. Do not replace unknown usage with invented zero-cost success. Forward cancellation where supported; provider cancellation and effect cancellation are different.

## Version checks

Use `natlang/runtime.py`, `natlang/tool_agent.py`, `natlang/host.py`, `ts-host/src/contracts.ts`, `ts-host/src/native/host.ts`, and `ts-host/src/browser/host.ts` as API anchors. The older `web/natlang_lite.mjs` helpers are useful trace/experimental utilities, not a substitute for the complete native browser interpreter.
