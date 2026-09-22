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
from natlang.files import FilesystemFileTree
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.invocation import RunOptions, SeedPolicy

# decoder is supplied by the embedding and implements natlang's chat contract.
def inspect_project(decoder, root_path):
    # inspect.nl declares: files: Dict<ProjectFile>
    root = load(Path("inspect.nl"), {
        "files": FilesystemFileTree(root_path),
    })
    runtime = Runtime(
        lambda lam: ToolAgent(decoder, validation_feedback="local"),
        options=RunOptions(seed=SeedPolicy("derived", 17)),
    )
    outcome, value = runtime.run_root(root)
    return outcome, value
```

`FilesystemFileTree` is a host representation of an ordinary, read-only
`Dict<ProjectFile>` argument. Natlang inspects it with `read_value` expressions
such as `files["docs"]["plan.md"]`; it does not gain a special namespace. Pass
the same `files` value positionally to a compatible natlang child that needs
it. `text_file_tree({"path": "text"})` supplies the corresponding
in-memory adapter. Both resolve branches and leaves lazily and cache each
observation for the lifetime of that value.

Inspect `outcome.kind` before using the value as completed. `load()` can interpret existing short string paths as files; use `load_definitions(entries, root_name, inputs)` for explicit in-memory values when that ambiguity matters. Checked definition entries use `args`, `returns`, exactly one of `instructions` or `code`, and optional `types`, `uses`, `effects`, `engine`.

`Runtime` supports declared capabilities and a registry of crisp executors. Consult `natlang/execution.py` for `CrispRequest`, the executor protocol, and portable value validation. An evaluator must report errors and return exact portable values; do not let arbitrary native objects silently enter the typed tree. Engine metadata should accurately describe environment lifetime, authority, and replay limits.

## Node

```ts
import { NatlangHost, NodeFileTree } from '@natlang/typescript-host';

export async function inspectProject(modelTurn, rootPath) {
  const host = new NatlangHost();
  try {
    const result = await host.run({
      // inspect.nl declares: files: Dict<ProjectFile>
      source: { kind: 'file', path: 'inspect.nl' },
      inputs: { files: new NodeFileTree(rootPath) },
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
`kind:'files'` contract. That source map defines the codebase; it is not
automatically a semantic argument. For host project data, pass
`textFileTree(files)` as a declared input in browsers and portable embeddings,
or `NodeFileTree(root)` in Node.

Application reducers that run once per event should construct filesystem trees
through the framework's `reducerInputs` factory. Reusing one `NodeFileTree`
object across runs also reuses its observation cache and can hide changes made
between events. A direct `host.run` call already has a single-run lifetime.

## Lazy `Dict<T>` providers

Use `LazyDict` with a `TreeProvider<T>` for other large keyed spaces. The
provider's `list(path)` returns immediate `{name, kind}` entries and
`read(path)` returns one typed leaf. The runtime presents this as the declared
`Dict<T>` and validates a leaf against `T` when it is observed. It never exposes
`LazyDict`, `TreeProvider`, or branch metadata as natlang types.

Good uses include repositories, document collections, asset metadata, build
inputs, trace collections, and read-only database projections where semantic
code needs to browse a few entries. Use crisp search or an indexed host query
when selection requires scanning the whole collection. Keep binary payloads in
the host and expose typed metadata plus targeted operations.

Avoid attaching a changing filesystem tree to a workflow whose declared
authority is a pinned repository, evidence collection, source revision, or
merge base. Either import the relevant file into that versioned input or make
the live provider and its identity part of the reproducibility contract.

Provider rules:

- Return stable, unique immediate names with `branch` or `leaf` kind. Reject
  traversal and symlink escapes in filesystem adapters.
- Assume each observed listing or leaf is cached once. Construct a new value
  when a run must see newer backing data.
- Bind it only to a compatible `Dict<T>` parameter. Normal eager dictionaries
  remain valid inputs for the same source.
- Do not pass the whole provider-backed value into a crisp function. Select a
  portable leaf first, or let crisp code use the native host API directly.
  Model-facing `eval` omits provider-backed fields from its portable execution
  view while retaining the other arguments.
- Do not expect `dump_state` or trace data to serialize the provider. Persist
  selected results and enough provider identity to reconstruct the binding.

For writes, give crisp code a scoped filesystem/database operation or let
natlang return a `{ path: Text, text: Text }[]` change plan. Python provides
`validate_file_writes` and `commit_file_writes` in `natlang.files`; Node exports
`validateFileWrites` and `commitFileWrites`; browsers export validation so the
owning application can choose its persistence mechanism. Commit helpers reject
absolute, parent, backslash, duplicate, and root-escaping paths and replace each
file atomically. A multi-file plan is not a transaction: report its receipts
and preserve partial-failure information. Do not make a read-only dictionary
secretly mutate its backing store.

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

Keep model/template quirks inside the adapter. Source programs use scope-eval-v1 tools and ordinary positional calls; adapters may map those tools to a provider's structured representation. Use the selected backend's supported typed tool representation. Do not silently unwrap malformed values or paste model-specific conventions into `.nl` files. Capture the exact presented tools/messages, not only the pre-adaptation request.

The shared OpenAI-compatible adapter retries one response whose tool arguments
are malformed JSON, adding a narrow correction to the original turn. A second
malformed response is a bounded transport failure. This repair happens before
runtime validation feedback, which handles well-formed calls that violate the
offered schema or current state.

When `max_tokens` is absent/null, omit a provider field that requires an integer. Report actual usage and termination reasons when available. Do not replace unknown usage with invented zero-cost success. Forward cancellation where supported; provider cancellation and effect cancellation are different.

## Version checks

Use `natlang/runtime.py`, `natlang/tool_agent.py`, `natlang/host.py`, `ts-host/src/contracts.ts`, `ts-host/src/native/host.ts`, and `ts-host/src/browser/host.ts` as API anchors. The older `web/natlang_lite.mjs` helpers are useful trace/experimental utilities, not a substitute for the complete native browser interpreter.
