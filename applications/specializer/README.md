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
policy.ts                 crisp rules and pluggable wrappers: targets, decline text, loop progress, guidance checks
promote.nl                promotion policy (promotionPolicy nl|shadow): promote, keep or demote a case or tier from its evidence
worthLooking.nl           target policy (targetPolicy nl|shadow): is a function worth a look now
summarizeDecline.nl       decline text (declinePolicy nl|shadow): the main obstacle and what would change it
chooseCondition.nl        directory reducer: one group's folder in, a measured condition out (or a reason to skip)
chooseCondition/semanticCheck   decision: is a condition how the decision is made, or does it need meaning
chooseCondition/sameApproach    decision: do two normalized programs do the same work
writeBody.nl              directory reducer: the chosen condition in, case.ts out
writeGuidance.nl          short positive guidance for the student, stored as the compilation's instructions.md (tier 2)
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

## Pluggable policies, split writer, guidance (2026-10-09)

- Store settings `targetPolicy` and `declinePolicy` (`crisp` default, `nl`, `shadow`; set with `natlang traces config`)
  select the crisp rule or `worthLooking.nl` / `summarizeDecline.nl`, as `promotionPolicy` selects `promote.nl`. Under
  `shadow` the crisp answer serves and the agreement is traced. The crisp bound of `minCalls` agent calls holds under
  every mode.
- A group's writer is two calls: `chooseCondition` returns a condition, the host measures it exactly (`group.measure`
  must show it valid, admitting at least one call of the group and none of another), then `writeBody` writes `case.ts`.
- The rounds loop is bounded by `--rounds` (a resource limit) and ends early when a round leaves every group's outcome,
  report and case unchanged (`madeProgress` in policy.ts: terminates by the limit, never stops while a group still gets
  new information, spends nothing on a round with identical input).
- Defaults of the CLI options and limits, each with its reason, are `DEFAULTS` in policy.ts; `--max-groups` sets the group cap.
- `writeGuidance.nl` distils the accepted groups' recorded calls into guidance (at most 1500 characters, every point an
  action). The crisp check rejects an over-long text or sentences written as prohibitions; one repair attempt, then no
  guidance. The text is stored as the compilation's `instructions.md`, which makes tier 2 (student plus guidance)
  appear in the ladder; it earns promotion through the shared evidence rule. `--no-guidance` skips it.

## Groups, spend, findings (2026-10-09)

The host splits a function's calls by what they did; each group (at least three training calls, at most eight groups)
gets a `chooseCondition` and then a `writeBody` call over a small folder (group.md, examples, others.md, report.md) with an exact `group.measure`
service. The cases are assembled, verified together, and a rejected case goes back to its group. Promotion also needs
`promotionLiveComparisons` (3) comparisons on live calls. `natlang compilations savings|findings|acknowledge|export-corpus`
show what serving saved against what specializing spent, what compiling found, and export cases and declines as a corpus.
`--max-busy` and `--idle-wait` make the specializer wait for an idle executor.
