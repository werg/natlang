# specializer: compile recorded calls into guarded crisp cases

Every natlang runtime records its calls in the machine's call store. The specializer reads those records and, for each
natural-language function that runs often, looks for invocation patterns that are already crisp in practice: inputs
of one kind for which the executor always ran the same code and produced the same kind of result. For each such
pattern it writes a case, a guard over the inputs and a TypeScript body, and stores the cases as the function's
compilation. Runtimes load compilations by themselves; the program's source is never changed. Design:
[plans/TRACE_SPECIALIZATION.md](../../plans/TRACE_SPECIALIZATION.md).

```sh
natlang specialize --definition support      # one function (name, source path or revision key)
natlang specialize --loop                    # every hot function, then shadow replays and audits; repeat
natlang specialize --dry-run --definition X  # verify without storing anything
natlang specialize --jobs-only               # only the pending shadow replays and audits
natlang compilations show support            # what was stored, how its cases are doing
```

`natlang specialize` is `natlang run applications/specializer --` with model options (`--profile`, `--provider`,
`--model`) passed to the run.

## Parts

```
main.ts                   crisp orchestration: targets, rounds, verification, storing, offline jobs
writeCase.nl              directory reducer: one group's folder in, case.ts out (or a reason to skip)
writeCase/semanticCheck   decision: is a condition how the decision is made, or does it need meaning
writeCase/sameApproach    decision: do two normalized programs do the same work
```

The crisp half lives in the runtime package (`ts-host/src/calls/`): `study` (examples, approaches with anti-unified
templates, induced guard conditions with exact precision), `renderEvidence`, `verifyCases` (each case replayed on the
recorded calls its guard admits, against the recorded service results, compared with the agent and judged where they
differ), `saveAccepted`, and `runJobs` (shadow replays and audits).

| Part | Decision | Why |
| --- | --- | --- |
| choosing targets | crisp | volume, standing declines and new calls since the last compilation are counts |
| grouping calls into approaches | crisp, with `sameApproach` | normalization and anti-unification are exact; whether two differently written programs do the same work is a judgment |
| finding conditions | crisp proposals, model decides | rule induction measures precision exactly; which conditions describe the function's real decisions is a judgment |
| semantic check | its own decision function | small models need the question asked on its own, with the instructions and examples |
| writing cases | the reducer | lifting a template, generalizing from examples and improving on the executor are authoring |
| declining | the reducer, or crisp when nothing repeats | a crisp decline costs no model call |
| verification and acceptance | crisp, with the runtime's judge | replay and equality are exact; better or worse is judged against the instructions, blind to which side is the case |

## Rounds

Each function gets up to `--rounds` (3) rounds. The reducer's folder holds `evidence/` (function, approaches with
examples, conditions, unclassified calls, history, last round's report) and `cases.ts`. After each round every case is
verified and the report goes back into `evidence/report.md`. Accepted cases are stored; cases that never pass are
dropped. When no case passes, or the reducer declines, the decline is recorded and the function is not looked at again
until its call volume doubles or its revision changes.

## Groups, spend, findings (2026-10-09)

The host splits a function's calls by what they did; each group (at least three training calls, at most eight groups)
gets one `writeCase` call over a small folder (group.md, examples, others.md, report.md) with an exact `group.measure`
service. The cases are assembled, verified together, and a rejected case goes back to its group. Promotion also needs
`promotionLiveComparisons` (3) comparisons on live calls. `natlang compilations savings|findings|acknowledge|export-corpus`
show what serving saved against what specializing spent, what compiling found, and export cases and declines as a corpus.
`--max-busy` and `--idle-wait` make the specializer wait for an idle executor.
