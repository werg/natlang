# Agent-facing surface and observed failures

> **Current interface (2026-09-24; [`spec/SPEC.md`](../spec/SPEC.md) is normative).** The model
> answers judgment work directly with `return_result` (status `success`, `blocked`, or `failed`), and
> uses a persistent TypeScript evaluation scope for exact work, where a top-level `return` stages the
> result and a reply without a tool call finishes. The other actions are `read_page`, `compact_history`,
> and `read_code`, `edit_code`, and `diff_code` when functions are imported. Directory
> reducers additionally receive file tools. A call keeps one conversation (the checkpoint continuations
> described below were removed). The observations below were recorded against earlier surfaces.
>
> Imported function sources have a fixed file set: their contents may be
> edited and imports relinked; creating, moving, and deleting function files is
> forbidden. Directory reducers receive a writable folder whose paths are
> relative to its root. `commit` selects changes and `folder.apply` retains
> them.
>
> The remainder of this file is historical inventory material from the
> pre-`scope-eval-v1` protocol. Its slash-path names and eight-tool lists
> describe old traces and are not the current agent interface.

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
| `tools_explicit.md` | Bonsai teacher adapter | A natural reply may finish after return and line closure | Positional path calls, explicit modes, `failed`, and `blocked` |

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
| `codebase/<function>` | Function instructions and associated source files | Existing-file content editable; file set fixed |
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

### `blocked`

```text
blocked(missing)
```

Ends without a result when required information is absent or no rule covers the case.

### `failed`

```text
failed(message)
```

Ends without a result when the instructions contradict themselves, require an invalid operation,
or demand an incompatible result. It is deliberately distinct from missing information even
though both quiesce the runtime.

## 4. Schema guidance and transport variants

There are three materially different presentations of the same conceptual tools:

1. **Native grammar.** Dynamic `x-natlang-alternatives` compile into a grammar. Syntax and many
   enum choices are forced, but runtime mode deliberately leaves value semantics fallible.
2. **Ordinary chat tools.** A backend receives JSON Schema through its native chat template.
3. **Bonsai teacher tools.** Tools-v4 uses names that do not collide with the server's call syntax.
   Only typed literal writes expand to numbered exact tools. Function calls keep one explicit tool
   per mode, limiting schema growth on larger codebases. Bonsai also receives a small reasoning
   budget and its own native template.

Optional adapters can encode write values as JSON text or remove changing path enums to improve
prefix-cache reuse. Those are transport choices, not runtime semantics. The current teacher
factory uses typed alternatives for literal writes; it does not use JSON-text values or
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

With local validation feedback, a model can repeatedly alter proposals until one passes, which
risks teaching it to revise requirements or fabricate a compatible value. Caller feedback makes
bad instructions and bad actions fail cleanly, but a single correctable slip also ends the
attempt. `failed` and `blocked` provide honest exits before
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

1. **Implemented in tools-v4:** split `call` into `run_function`, `for_each`, `fold`, `repeat`, and `resume`.
2. **Implemented in tools-v4:** split literal write, value copy, and function copy.
3. Remove the redundant type argument wherever the destination or selected operation determines
   it; retain an explicit type only when creating a genuinely new local with no other type source.
4. **Implemented in tools-v4:** all call inputs are positional workspace paths; literals must first
   be written as typed locals. Each mode states which leading positions it supplies.
5. Keep natural completion, caller-level validation, explicit error/blocker exits, fresh-context
   continuations, and fallible semantic proposals.
6. Materialize teacher and synthetic IR into the selected surface so backend aliases and schema
   versions remain migrations rather than permanent facts in the corpus.

This preserves model control over ordering and interpretation. It changes the vocabulary used to
state an action; it does not add deterministic orchestration of the program.
