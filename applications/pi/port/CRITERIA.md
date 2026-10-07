# How to classify each unit of pi for the natlang port

Background. natlang is a language whose functions are natural-language instructions (`.nl` files with typed args and
returns) executed by a language model. The model runs a TypeScript eval scope to compute exactly and to call the
functions and services it can reach. The owner wants a *port* of pi's harness, based on **pi-durable**
(`/home/werg/src/pi/packages/durable`, a durable agent harness; its normative design is `docs/spec.md`, and the coding
agent's frontend on it is `packages/coding-agent/src/experimental/durable`): go through the code unit by unit, function
by function, and translate each into natural-language pseudocode. Use the spec to understand intent and invariants;
cite spec sections where they define a unit's rules. The executors are **low-capability, fast
models**, so detail is desirable: instructions should spell rules out, and units should stay small enough for a small
model to carry out reliably.

For every unit (function, method, significant closure such as a tool's execute function) decide one of:

- **fn** — its own natural-language function with a typed contract. Choose this when the part has its own data and
  contract, can be checked or retried on its own, runs per item in parallel, needs a scored decision, or would
  overload one call if interleaved with its neighbours. Name it (kebab or camel case as pi names it).
- **inline** — one instruction (or a few sentences) inside its caller's natural-language function. Choose this when it
  is tightly bound to the caller's state and a small model does it reliably in one step. Name the caller.
- **implicit** — left to the model with no instruction (rare: only what small models do reliably, like JSON
  parse/stringify, sorting, basic arithmetic in eval). Anything with rules (pi's exact truncation format, path
  resolution rules) is NOT implicit.
- **crisp** — a TypeScript helper in a callable folder: straightforward deterministic plumbing (a diff, a byte count
  truncation, a path normalization), which a natural-language function calls.
- **service** — the outside world or durability infrastructure, implemented in the host: provider HTTP (via the pi-ai
  library), process spawning, file system, terminal, clock, and the storage/commit machinery. Natural-language
  functions call services through declared interfaces.
- **out** — not ported, with the reason (TUI rendering, telemetry, OAuth flows, package management, out of scope).

Do not go below what matters: no entries for trivial getters/setters, re-exports or one-line wrappers unless they
carry a rule; group such trivial members in one row ("getters/setters for X: crisp state record"). Classes become a
typed state record plus functions over it; say what the state record holds.

Where the line falls for pi-durable: durability machinery (the atomic commit line, storage backends, Chord document
synchronization, drafts, cancellation contexts, watchers) is host infrastructure: service or crisp, described by the
interface the harness logic uses. The harness's behaviour is natural language: what a turn does, tool rounds and tool
execution rules, compaction policy and summaries, context and system prompt assembly, hook and extension semantics as
they affect a run, subagents and child tasks, submission handling, scheduling decisions that are policy (what runs
next, what a busy conversation does with new input). When a unit mixes both, say how it splits.

natlang facts that affect decisions:
- A named function can call only the functions in its own companion folder (`foo/` beside `foo.nl`) and services.
  Shared code across modules must be a service, a crisp helper, or passed in as data. Note each unit's callers and
  callees so the layout can be decided.
- Loops in eval are finite (`for...of`, array methods); open-ended repetition uses `iterateOn(step, state)` with a
  limit and a stopping predicate. Recursion is structural only.
- `decide(fn, ...args)` returns a finite judgment's probabilities (for thresholds).
- Directory reducers work on a copy of a folder and keep changes only on success (`folder.apply`). natlang also has
  `EventLoop` reducers (`reduce(state, event) => state`, serial application, duplicate suppression, commit before
  publish); note where these could carry a pi-durable mechanism instead of porting it.
- Events/streams/async iterators/abort signals have no direct equivalent: say how the unit's behaviour maps (a returned
  record, a service callback, a log line), or mark it service/out.

Output format (markdown), per file:

## <path relative to packages/> — <one sentence: what this module is for in pi>

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|

After the table, a short "Notes" paragraph: cross-module dependencies, state, anything that needs an owner decision.
Be concrete in "what it does": a reader should be able to write the pseudocode from it (include the exact rules,
limits, formats and messages the unit uses). Read the code fully; do not guess from names.
