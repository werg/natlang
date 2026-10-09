---
description: A repository migration in natural language. Understand the request, survey and classify every site that mentions the old thing, plan, write and check one patch per site, apply them, run the repository's checks, and repair from the failures until the checks pass or the work is settled.
args:
  request: string
  snapshot: RepoSnapshot
  checks: string[]
  seeds: string[]
  attempts: number
uses: [site/classify, site/edit]
returns: Migration
---
Migrate the repository as request says, with the stages in your folder. snapshot is the base revision (its files are
the manifest), checks are the ids of the repository's checks, seeds are extra strings to search for, and attempts is
the number of repair rounds allowed after the first candidate. The repository service holds the revisions; it never
writes the original checkout. Work in eval and return the result as an object built from the values below.

1. Understand. intent = understand(request, snapshot, checks, seeds).
2. Survey. sites = survey(intent, snapshot.revision). usages = await Promise.all of classify(intent, site) for every
   site. classified = the sites paired with their usages: { site, usage }.
3. Plan. planned = plan(intent, classified).
4. Edit. edits = await Promise.all of edit(intent, c, snapshot.revision) for every c in classified whose site id is in
   planned.edits. Add { site: the id, usage, patches: [], status: "left", note: the reason } for every entry of
   planned.leave. patches = the patches of all edits, in the order of planned.edits.
5. First candidate. state = { snapshot, validation: null, rejected: "", findings: [], remaining: attempts,
   revisions: [snapshot.revision] }. When patches is empty, rejected = "no site produced a patch: " and the notes of
   the failed edits. Otherwise try: candidate = repository.apply(snapshot.revision, patches), then validation =
   await repository.validate(candidate.revision), and state = { ...state, snapshot: candidate, validation,
   revisions: [snapshot.revision, candidate.revision] }. When apply rejects, rejected = the error's message.
6. Repair. stop = settledCrisp when repository.implementation("settled") is "crisp", else settled. final = await
   round.iterateOn(state, intent).withMeasure(s => s.remaining).until(stop). When the loop ends with an error whose
   name is "IterationLimitError", final = the error's lastState: the repair budget is spent and the last candidate is
   the honest result. Any other error is raised.
7. Report. validation = final.validation, or await repository.validate(final.snapshot.revision) when it is null.
   report = repository.report(final.snapshot.revision, validation). closing = summarize(intent, planned, edits, final,
   report).
8. Return { report, intent, plan: planned, edits, findings: final.findings, rounds: attempts - final.remaining, summary:
   closing.summary, next: closing.next }.
