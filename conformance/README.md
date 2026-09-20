# Conformance suite

Program references and structured tests that define what "the interpreter works" means
for natlang `spec/SPEC.md` v0.2.

## `programs/`: interpreter conformance

Each file is one program with unambiguous toy inputs.

| Key | Meaning |
|-----|---------|
| `program` | the root lambda in the serialization of SPEC §11, optionally with an inline `codebase` |
| `program_file` | instead of `program`: the entry function of a code base on disk (`../../examples/triage/main.nl`) |
| `start_state` | used instead of `program` when a test begins from a partly reduced tree |
| `inputs` | values bound to the root's `args` |
| `streams` | items fed to an open list; `$close` closes it |
| `expect.value` | the exact final value |
| `expect.checks` | properties, when the value is not unique: `crisp` (TypeScript over `value`) or `judge` (a yes/no question put to a strong model, thinking off) |
| `expect.status`, `note_checks` | for programs that must end in a blocker |
| `expect.emitted`, `journal_entries` | expected effects |
| `lint.must`, `must_not`, `may`, `max_actions` | what a conforming **trace** does, independent of the final value |
| `reference` | the reference trace, in the tools-v2 (SPEC §5) vocabulary: see below |

### `reference`

A mapping from function name (the root's `function:`, or a code-base
function's name) to how a good interpreter runs it:

- **a function whose pseudocode calls others** gets `calls`: an ordered list
  of `{tool: <name>, args: {...}}`, one per turn. `tool: glue` is shorthand for
  a `run_code` whose result is then `write`-ten: `{tool: glue, code, path, type}`.
- **a natural-language leaf** gets `answer` (one instance) or `answer_by` (a
  map keyed on the value of its `item` parameter if it has one, else its first
  parameter — the key is the value itself for Text, else
  `json.dumps(value, sort_keys=True)`), or `blocker` (it should end in
  `report_blocker`). An `answer`/`answer_by` entry may instead be
  `{blocker: "...", then: <value>}`: quiesce with that blocker the first time
  this pending node runs, and write `then` when it is resumed (map partial
  repair). A leaf may give `variants: [{if_body_contains, answer}, ..., {answer}]`
  to answer differently depending on its (possibly edited) instructions text,
  for copy-edit-call scenarios.
- `known_issue: "..."` marks a program whose reference cannot be driven
  through the current harness (with the reason); its test is skipped.

`tests/test_conformance_reference.py` runs every program that has a
`reference` block through the real harness with a small scripted agent that
follows it, and grades the outcome with `natlang/checks.py`.

Grading is `natlang/checks.py`: outcome (`yes` / `no` / `?` when a judge check
could not be evaluated) and, separately, the structure the run used. A correct
value reached by doing a callee's work by hand, by one call per item, or by
guessing where a blocker was due is a failure of the trace.

Two kinds of program:

- **Leaf programs** (01, 02, 16, 18, 21, ...): one judgment, extraction or
  decision with no code base. They test the prompt-like capability that every
  code base rests on.
- **Code-base programs** (23 onwards): pseudocode with functions, calls over
  lists, exact glue, conditionals, locals, nested code bases. Their `lint`
  rules are checkable because the required structure is in the program text.

Programs 03–22 are code-base programs (pseudocode plus an inline `codebase`,
or `program_file` for one on disk), each with a `reference` block. Program 20 keeps its structure but
its `reference` is a `known_issue`: a root-level `Fold` over an open,
streaming list has no supported loading path in the current harness.

Uses: acceptance test for the harness with a strong model as interpreter;
test of the language with the teacher (`TRAINING.md` §3.4); seed for the
synthesizer's shapes; permanent regression suite.

Structured tool checks live in `tests/test_structured_session.py`, while
`tests/test_conformance_reference.py` runs the program references, including
combinators and resumed tasks.

## Checking

```
python3 tools/check_conformance.py
```

parses every file, parses every type expression against the SPEC §2 grammar,
resolves named types, and checks the structure of every pending-node wrapper.
