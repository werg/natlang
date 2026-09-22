# Agent-facing surface and observed failures

This document inventories the surface presented to a natlang interpreter model as of
2026-09-22. It distinguishes what the model literally receives from harness behavior that
shapes the episode without appearing as a tool.

## 1. Interaction envelope

An interpreter episode contains these channels, in order:

1. **System message.** One of the tool-use prompts described below.
2. **User request.** The pseudocode, its required return type, and the signatures of callable
   functions. Programs with source line marks are rendered with `[ ]`, `[x]`, and `[-]`.
3. **Injected opening tool call.** The harness normally inserts an assistant `read({path:
   "args"})` call on the model's behalf.
4. **Opening tool result.** A workspace summary: inputs, existing locals, return status,
   continuation note, and recent effect-journal entries.
5. **Assistant turns.** Zero or more native tool calls, optionally in ordered batches.
6. **Tool results.** Each proposal receives its own result. A batch is not atomic.
7. **Terminal assistant turn.** A turn with no tool call asks the runtime to finish. The text is
   retained as an audit note; it is never parsed as the function result.

The actual result is only the typed value stored at `return`.

### System-prompt profiles

| Profile | Used for | Completion wording | Notable differences |
|---|---|---|---|
| `tools_delegate.md` | Live delegated interpreter/student style | End without text after a valid return | Concise operational rules; explicitly discourages unnecessary reads |
| `tools_small.md` | Reference generation and TS parity | End without text | More examples, explicit ordered-batch semantics |
| `tools_teacher_compact.md` | Bonsai teacher adapter | A natural reply may finish after return and line closure | Explains `report_error` and `report_blocker`; compact teacher-specific wording |

This means teacher and student currently receive semantically close, but not byte-identical,
instructions about the final assistant reply.

## 2. Workspace namespace

The model refers to data through slash-separated paths:

| Namespace | Meaning | Mutability |
|---|---|---|
| `args/<name>` | Declared function inputs | Read-only |
| `args@effects` | Complete external-effect journal | Read-only |
| `let/<name>` | Model-created intermediate value | Writable |
| `return` or `return/<field>` | Required result and its fields | Writable |
| `codebase/<function>` | Function instructions | Read-only |
| `let/<copy>/instructions` | Instructions of a copied function | Editable |

The opening workspace view previews small values and names long values for later `read`. It also
shows which return fields are filled or missing and which pending subtasks exist. With the
experimental `state_view` option, every successful result is followed by the full workspace,
current marks, and function list. That option is off in normal use; earlier probes found that the
extra context could reduce success.

At most 48 paths are used in many dynamic schema enums. Named values are ranked ahead of list
elements. Lazy host dictionaries may accept paths beneath a known prefix.

## 3. Tools shown to the model

The conceptual list is stable during an episode, while schemas are rebuilt from current state.
`call` is omitted when no functions exist, and `mark_done` is omitted when line marking is not
active or no marked work remains.

### `read`

```text
read(path, start?, end?)
```

Reads a workspace value, a function body, a long-value range, or the effect journal. The prompt
tells the model that a path can be passed to a function without reading the value first.

### `write`

```text
write(path, type, value?, source?, done?)
```

This single tool performs three distinct operations:

1. Write a literal `value`.
2. Copy an existing value from `source`.
3. Copy a named function by supplying `type="Function<name>"` with neither value nor source.

`path` may be an existing writable slot or a new `let/<name>`. `type` repeats the destination
type even when the destination already determines it. `value` is exposed as a union of currently
writable shapes plus an unconstrained new-local shape. The runtime permits semantically invalid
proposals so they can be rejected honestly instead of being made impossible by decoding.

`done`, when enabled, is a line number or inclusive two-element range applied only after a
successful write. It is line-marking shorthand; it is unrelated to episode termination. There is
no terminal `done` tool.

### `edit`

```text
edit(path, old, new)
```

Performs an exact, once-only text replacement. The model cannot edit its own current program.
Its main supported use is adapting `let/<copy>/instructions` after copying a function. `old` must
match exactly and occur once.

### `run_code`

```text
run_code(code, engine?)
```

Runs exact glue work over `args` and `locals`, such as arithmetic, counting, sorting, and string
operations. The expression's value is returned to the model; a later `write` stores it.

The surface label is `tools-v2` when the runtime supplies the historical implicit engine. It is
`tools-v3` when `engine` is required and enumerates the available executors. This version change
does **not** split `call` or `write`; it only makes code-engine selection explicit. The TS runtime
requires `engine="typescript-host"`. Historical v2 traces can be projected by inserting the
chosen engine in `run_code` calls.

### `call`

```text
call(function, to, inputs?, values?, over?, init?, until?, max?, done?)
```

One tool encodes five operations:

| Operation | Distinguishing fields | Meaning |
|---|---|---|
| Invoke once | `function`, `to`, optional `inputs`/`values` | Call a function once |
| Map | `over` | Call once per list item; the omitted parameter receives each item |
| Fold | `over`, `init` | Carry `acc` through the list; runtime supplies `acc` and `item` |
| Repeat | `init`, `until`, `max` | Apply a transition until a Boolean check succeeds |
| Resume/retry | only `function`, `to` for a pending destination | Retry an unfinished child |

`inputs` maps parameter names to **workspace paths**. `values` maps parameter names to typed
**literal values**. They may be mixed only when their parameter sets are disjoint. For map and
fold, the runtime implicitly supplies some parameters based on omitted bindings. For repeat,
`init` is a workspace path despite the broad top-level schema also admitting literal JSON shapes.

The schema contains mode-specific alternatives for constrained decoding. Ordinary chat backends
still see one top-level tool with many optional fields unless an adapter expands those
alternatives into separate backend tool names.

### `mark_done`

```text
mark_done(start, end?, skipped?)
```

Closes one line or an inclusive line range. `skipped=true` records that a branch did not apply.
Blank lines and other non-substantive lines are not completion obligations. A valid return plus
all substantive lines closed permits the next natural end of turn to finish.

### `report_blocker`

```text
report_blocker(missing)
```

Ends without a result when required information is absent or no rule covers the case.

### `report_error`

```text
report_error(message)
```

Ends without a result when the instructions contradict themselves, require an invalid operation,
or demand an incompatible result. It is deliberately distinct from missing information even
though both quiesce the runtime.

## 4. Schema guidance and transport variants

There are three materially different presentations of the same conceptual tools:

1. **Native grammar.** Dynamic `x-natlang-alternatives` compile into a grammar. Syntax and many
   enum choices are forced, but runtime mode deliberately leaves value semantics fallible.
2. **Ordinary chat tools.** A backend receives JSON Schema through its native chat template.
3. **Bonsai teacher tools.** `call` is aliased to `call_function`, because that server cannot emit
   a tool literally named `call`. Typed alternatives expand `write` and `call` into many numbered
   backend tool variants, then map them back to the conceptual names after generation. Bonsai
   also receives a small reasoning budget and its own native template.

Optional adapters can encode write values as JSON text or remove changing path enums to improve
prefix-cache reuse. Those are transport choices, not runtime semantics. The current teacher
factory uses typed alternatives and the `call_function` alias; it does not use JSON-text values or
cache-stable schemas.

Only simple coercions happen at execution: a single JSON string layer can be parsed for a
non-Text slot, and scalar Boolean/number/null values can become JSON text for a Text slot. The
runtime does not unwrap compatibility objects such as `{ "value": 17 }`.

## 5. Episode control that is not an ordinary tool

### Natural completion

A no-tool assistant turn triggers validation. Completion succeeds only when `return` is present
and type-valid and all substantive marked lines are closed. Missing work produces at most two
small line-number nudges. The final prose reply is only an audit note.

### Validation feedback

In `caller` mode, the first rejected proposal ends that lambda attempt with a structured
validation failure for the parent caller. The same interpreter is not invited to repair or fudge
it. In `local` mode, the error is appended as a tool result and the model can try again. Teacher
collection and current readiness evaluation use caller-style failure for the behavior we want to
train, although some application evaluation scripts can explicitly select local mode.

### Continuations

The default agent checkpoints after 6 assistant turns or 12 messages when unfinished work is at a
safe boundary. It asks for a note of at most 800 characters, saves that note in runtime state,
and starts a fresh conversation containing the current workspace and note. Previous conversation
messages are never pasted into the new segment. The limit controls the number of items in a
training trajectory segment rather than truncating a program by token count.

### Careful review

An optional fork can review a low-confidence write or a structurally broad action before it is
executed. The review sees the exact proposal and must call:

```text
review_write(decision = approve | withdraw | error | blocker, reason)
```

The original session continues unchanged on approval; withdrawal allows one replacement under
the configured policy. Confidence comes from token probabilities at `write.value` or `edit.new`.
It is a trigger signal, not a calibrated probability of semantic correctness. Repeating too much
state or instruction text in the review has previously reduced success.

### Budgets

Normal runtime model turn, episode token, episode turn, and wall-clock budgets default to
unbounded. Evaluation scripts may impose explicit limits. Continuation thresholds remain enabled
by default because they reset conversation context without failing the underlying task.

## 6. Python and TypeScript parity

Both runtimes expose the same workspace model, dynamic read/write/edit/call/marking/error tools,
caller versus local validation, review fork, natural completion, and fresh-context continuations.
The checked parity target is currently called `tools-v3` because both runtimes make the execution
engine explicit. The main intentional difference is the available engine: Python can expose
configured executors, while the native TS host exposes `typescript-host`.

Model-specific template adaptations, aliases, typed-alternative expansion, and log-probability
collection live in the Python decoder/teacher path rather than the TS runtime itself.

## 7. Failure taxonomy

### 7.1 Operation-mode ambiguity in `call` — primary failure

The model commonly chooses the correct function and destination but expresses the argument
relationship through the wrong optional field. The clearest repeated error is:

```text
let/labels/args/row: expected Joined, got Joined[]
```

The intended operation is a map: `over=let/joined`, with each `Joined` item implicitly bound to
`row`. The model instead places the entire `Joined[]` path in
`inputs={"row":"let/joined"}`. Both forms are syntactically plausible members of one large call
schema, but they mean different cardinalities.

Related errors include binding the same parameter in both `inputs` and `values`, supplying a
checker function where a state value is required, and using the repeat fields as if `init` were a
literal or destination. These are evidence that the overloaded action representation is itself a
bottleneck.

### 7.2 Dependency and turn-order errors

The model sometimes batches or advances past a result it has not received. In the saga cases it
tries to use `let/kind` or `let/next` before the call that creates that local has completed. One
failure even treats `read_event` as a workspace path. This violates the explicit rule that a
dependent action must wait for the next turn.

### 7.3 Workspace path construction errors

The model invents plausible-looking paths that the workspace does not contain, for example
`let/initial/state`, or tries to edit a nested structured value as text. It may append a field to
make an incompatible type appear compatible. Dynamic enums prevent some of these mistakes in a
strict native grammar, but chat-server schemas and runtime-mode proposals can still admit them,
and a path can be legal text while semantically wrong.

### 7.4 Literal-versus-reference confusion

The distinction between `inputs`, `values`, `write.value`, and `write.source` is easy to lose:

- a workspace path may be emitted as literal text;
- a literal may be placed in the path-only `inputs` map;
- a function or pending child may be used where its eventual result is required;
- a value may be copied by re-emitting it instead of using `source`.

The repeated type field on `write` adds another decision even when the destination already fixes
the type.

### 7.5 Exact-edit failures

`edit` requires byte-exact text that occurs once. Models sometimes guess a structural path or an
`old` fragment without first reading the editable instructions, producing `old-not-found`. This
is partly a legitimate precision requirement and partly a discoverability problem: structured
data changes and instruction-text edits look similar at the path level but only the latter belongs
in `edit`.

### 7.6 Over-reading and action thrashing

Earlier checkpoints read list elements and nested fields one by one even when a whole workspace
path could be passed directly to `call`. This grows context, loses prefix-cache reuse as dynamic
schemas change, and can exhaust explicit evaluation budgets. Prompt changes reduced this, but the
surface still offers many readable paths and regenerates schemas after state changes.

### 7.7 Marking and branch closure errors

Models sometimes mark a rejected branch as done instead of skipped, include an untaken branch in
an inclusive `done` range, or leave a control-flow line open after producing a correct value. This
is a finer bookkeeping error than returning the wrong value. The runtime distinguishes completed
work from skipped work, ignores blank/non-substantive lines, and still rejects an invalid return
even when every line is closed.

### 7.8 Premature natural completion

A model may end its turn with a missing/partial return or open lines. The harness catches this.
It sends limited mark-only nudges for open lines and otherwise returns a validation failure. There
is no need for a separate done action, but the model must learn that silence is meaningful only
after the workspace is complete.

### 7.9 Error honesty versus repair pressure

With local validation feedback, a model can repeatedly alter proposals until one passes, which
risks teaching it to revise requirements or fabricate a compatible value. Caller feedback makes
bad instructions and bad actions fail cleanly, but a single correctable slip also ends the
attempt. `report_error`, `report_blocker`, and optional careful review provide honest exits before
execution, although the model still needs data showing when to use each one.

### 7.10 Guided-generation commitment

Grammar guidance prevents malformed tool syntax and can restrict choices to state-derived
alternatives. Once the model starts a particular alternative, later tokens can be forced into
that mode. This is useful for syntax but harmful when the early choice was `inputs` and the model
really needed `over`. Runtime write constraints intentionally allow wrong semantic proposals, yet
the overloaded schema still turns an early mode choice into a large commitment.

### 7.11 Transport drift

The model-facing surface is not literally identical across backends. Bonsai sees aliases and many
numbered typed variants; a native-grammar model sees one conceptual name with constrained
alternatives; an ordinary chat model sees broad optional properties. A behavior learned against
one serialization may not transfer cleanly to another even when all calls map to the same runtime
operation.

### 7.12 Training-distribution failures

The recent LFM experiments exposed three separate data issues:

- exact deduplication reduced already-rare action targets until balancing was added;
- canned reasoning and very short targets allowed low token loss without correct program
  execution;
- teacher-forced single turns did not cover enough of the states reached by the model's own
  mistakes.

Action-only, balanced, and on-policy corrective passes improved diagnostic turn accuracy but did
not solve whole programs. This supports simplifying the action surface rather than adding more
prompt text around the same representation.

## 8. Current evidence

The best balanced LFM checkpoint reached **1/9 whole programs** and **7/12 diagnostic turns**.
An exact/on-policy corrective pass evaluated on held-out seeds reached **0/9 whole programs** and
the same **7/12 diagnostic turns**.

Across the 18 whole-program attempts in those two evaluations:

- all six reconciliation runs made the same map-versus-scalar binding error;
- five dependency runs failed through a wrong value/function binding, duplicate binding,
  invented field, or invalid exact edit; one dependency run succeeded;
- all six saga runs violated a dependency/path boundary around the locals produced by
  `read_event` and `transition`.

The failures are highly structured and repeat across unseen seeds. More examples remain useful
for sequencing and honest exits, but the evidence says that training alone should not carry the
full burden of distinguishing the current overloaded modes.

Evidence files:

- `runs/evaluations/lfm25-8b-live-action-balanced-v2-step600/readiness.json`
- `runs/evaluations/lfm25-8b-live-action-balanced-v2-step600/applications.json`
- `runs/evaluations/lfm25-8b-onpolicy-actions-heldout31/readiness.json`
- `runs/evaluations/lfm25-8b-onpolicy-actions-heldout31/applications.json`

## 9. Refactor implications

The inventory points to a smaller decision at each action site:

1. Split `call` into explicit invoke, map, fold, repeat, and resume operations.
2. Split literal write, value copy, and function copy.
3. Remove the redundant type argument wherever the destination or selected operation determines
   it; retain an explicit type only when creating a genuinely new local with no other type source.
4. Make path references a visibly different schema concept from literals, with mode-specific
   parameter requirements and no implicit “one omitted parameter” puzzle.
5. Keep natural completion, caller-level validation, explicit error/blocker exits, fresh-context
   continuations, and fallible semantic proposals.
6. Materialize teacher and synthetic IR into the selected surface so backend aliases and schema
   versions remain migrations rather than permanent facts in the corpus.

This preserves model control over ordering and interpretation. It changes the vocabulary used to
state an action; it does not add deterministic orchestration of the program.
