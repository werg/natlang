# migration: repository migrations in natural language

A request ("rename sum to add") becomes a reviewable candidate. Every decision is a natural-language stage:
understand the request, survey every site that mentions the old thing, classify each usage, plan, write one exact
patch per site, check each patch, apply them, run the repository's checks, triage failures, and repair in a bounded
loop with a natural-language stopping judgment. Crisp code is the fixed order of the stages (the driver in `index.ts`),
the outside world (the `repository` service) and the command line. [DECOMPOSITION.md](DECOMPOSITION.md) records the decision for every part.

```
index.ts migrate          the driver (TypeScript): understand, survey, classify, plan, edit, first candidate, repair loop, report
migrate/understand.nl       planning from the manifest: old and new thing, search queries, invariants
migrate/survey.nl           search evidence merged into sites (one per place)
migrate/plan.nl             which sites to edit and in what order, which to leave, risks
migrate/round.nl            one repair round: triage, repair, apply (a rejection is data), validate
migrate/settled.nl          the loop's stopping judgment (decision)  pluggable: or settledCrisp.ts
migrate/summarize.nl        closing words for the reviewer
site/locate.nl              one hit widened to the statement around it, exact lines from the service
site/classify.nl            the usage pattern (declaration, call, import, ...) and edit or leave
site/edit.nl                one site: write the patch, check it, write it again once with the problem
  edit/patch.nl               the exact old/new snippets
  edit/exact                  is the patch exact and in scope   pluggable: crisp counts or exact/judge.nl
loop/triage.nl              failing check output or a rejected patch to findings with kind, place, evidence
loop/repair.nl              findings to patches, with the same site units
```

## State model

Stages decide on immutable revisions and return data (`Intent`, `Site`, `Usage`, `Plan`, `Patch`, `Finding`).
Applying patches is one service call that yields a new revision or rejects with a message; a rejection is stored in
`RepairState.rejected` for the next round. The loop is `round.iterateOn(state).withMeasure(s => s.remaining)` with
`settled` as the stopping predicate: the measure ends a loop whose judgment never says stop, and the last candidate is
then reported as it is. Derived values (sites from hits, usages from sites, the plan from usages) are never stored
twice, and nothing loops without a bound.

## Running

```sh
natlang run applications/migration -- manifest.json [--attempts 3] [--seed TEXT] [--exact crisp] [--settled crisp]
```

`manifest.json` is `{ "root": ".", "request": "...", "files": ["a.mjs"], "checks": [{ "id": "t", "argv": ["node", "--test"] }] }`.
The checkout is never written: candidates live in memory and are materialized in a temporary directory only to run
the checks. The report lists the changed files with digests and the check results.

## Tests

`ts-host/test/repository-migration.test.mjs` runs every stage with a scripted model that answers with the algorithm the
stage's instructions spell out, against real checks: repair from check output, a complete first candidate, the exact
retry with the problem, a spent budget, unrepairable findings, the pluggable settings, and the service's exact
operations.

## Program migration (`--target program`)

`program/` migrates one legacy prose program (a design brief, an Astra/Sol application) into semantic FP, or deletes it
(`plans/semantic-fp/07_CORPUS_MIGRATION.md`; the "Program migration" section of [DECOMPOSITION.md](DECOMPOSITION.md)).
`program/host.ts` is the fixed sequence and the exact checks; the decisions are typed judgments beside it:

```
program/host.ts           findings from natlang check, keep or delete, step kinds, rewrites under semanticRepair
program/worthKeeping.nl     migrate or delete (decision; a program with no passing original case is deleted first)
program/stepKind.nl         judgment, mechanism, guard-for-type or redundant, per step (decision)
program/rewriteLeaf.nl      the new instructions of a leaf judgment
program/rewriteDriver.nl    a driver's fixed steps as a TypeScript helper, and the sentence that calls it
program/nextAttempt.nl      retry or give up after a rejected equivalence comparison (decision)
```

```sh
natlang run applications/migration --target program -- propose UNIT.json OUT_DIR
natlang run applications/migration --target program -- settle SETTLE.json
```

`scripts/semantic_fp/migrate_corpus.py` runs it over a corpus: units from a baseline report, `propose` in parallel on
the served model (or a teacher with `--provider`), the equivalence measurement of the candidates
(`scripts/semantic_fp/equivalence.py`), acceptance by the 07 §4.2 rule, `settle` for rejections, and the commit and
packaging of the accepted pairs.
