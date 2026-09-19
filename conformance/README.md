# Conformance suite

Programs and harness scripts that define what "the interpreter works" means
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

Programs 03–15, 17, 19, 22 were written as prose for the earlier surface and
carry `canonical_trace`s in the harness's trace notation (SPEC Appendix A).
They still run and are graded on outcome; **they are to be rewritten as
code-base programs** (each construct as pseudocode plus the functions it
names), at which point their `canonical_trace` and `lint` keys are replaced.

Uses: acceptance test for the harness with a strong model as interpreter;
test of the language with the teacher (`TRAINING.md` §3.4); seed for the
synthesizer's shapes; permanent regression suite.

## `harness/`: harness conformance

Scripted operation sequences with the harness's required responses, in the
trace notation of SPEC Appendix A. No model is involved; where child behaviour
matters a `stub_agent` stands in. `decoding: unconstrained` means the script
deliberately sends operations the grammar would normally make impossible, to
test the validator behind it.

## Checking

```
python3 tools/check_conformance.py
```

parses every file, parses every type expression against the SPEC §2 grammar,
resolves named types, and checks the structure of every pending-node wrapper.
