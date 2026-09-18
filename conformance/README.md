# Conformance suite

Small programs and harness scripts that define what "the interpreter works"
means for natlang `spec/SPEC.md` v0.1. Organized by language construct, not by
application domain.

## `programs/`: interpreter conformance

Each file is one program with unambiguous toy inputs.

| Key | Meaning |
|-----|---------|
| `program` | the root pending node, in the serialization of SPEC §11 |
| `start_state` | used instead of `program` when a test begins from a partly reduced tree |
| `inputs` | values bound to the root's `in` |
| `streams` | items fed to an open list; `$close` closes it |
| `expect.value` | the exact final value |
| `expect.checks` | properties, when the value is not unique: `crisp` (TypeScript over `value`) or `judge` (a yes/no question put to a strong model) |
| `expect.status`, `note_checks` | for programs that must quiesce |
| `expect.emitted`, `journal_entries` | expected effects |
| `lint.must`, `must_not`, `may`, `max_actions` | what a conforming **trace** does, independent of the final value |
| `canonical_trace` | the reference policy's action sequence, `>>>` action, `<<<` result (abridged) |

A run passes when the expectation holds **and** the trace passes lint. A
correct value reached by unrolling a loop or by leaning on scratch state is a
failure.

Uses: acceptance test for the harness with a strong model as interpreter
(Phase 1); teacher selection (`TRAINING.md` §3.4); seed for the reference
policy and the first synthetic programs; permanent regression suite.

## `harness/`: harness conformance

Scripted action sequences with the harness's required responses. No model is
involved; where child behaviour matters a `stub_agent` stands in.
`decoding: unconstrained` means the script deliberately sends actions the
grammar would normally make impossible, to test the validator behind it.

## Checking

```
python3 tools/check_conformance.py
```

parses every file, parses every type expression against the SPEC §2 grammar,
resolves named types, and checks the structure of every pending-node wrapper.
