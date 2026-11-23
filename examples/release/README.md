# Release desk

Writes a software release with the convergence and workspace combinators of `@natlang/node/std`. TypeScript holds the
changes, the exact rules and the verifiers; every judgment is a typed natural-language function.

```sh
natlang run examples/release        # notes, summary, merged release record, release section and announcement
natlang traces show CALL_ID         # what one call did; std:<combinator> traces list each stage's calls
```

## Files

- **types.ts**: the data. A change's `description` is `Untrusted<string>`; the summary is `Is<string, "...">`.
- **notes.ts**: `semanticRepair(first, { check, fix, attempts: 3 })`: the exact `notesProblems` check feeds an inline
  `nl` fix until every change has its line and no line is too long. `fixedPoint(revise, first)` revises the summary
  until the shipped `settled` judgment says a revision only reworded it.
- **merge.ts**: `semanticMerge(base, ours, theirs, { check })` over the release record. Fields only one side changed
  are taken from it; fields both changed are judged by the shipped `intentOf`, `compatible` and `combine`, and the
  version must still be a version number. What does not merge is listed in `unresolved`.
- **publish.ts**: `proposeAndVerify(folder, writeSection, { verify, attempts: 2, args })` writes `releases/VERSION.md`
  on a private copy and accepts it only when the exact `sectionProblems` verifier passes; a failed draft is proposed
  again with the problems. `branchAndSelect(folder, branches, { verify, goal })` writes the announcement in two
  styles, each on its own copy, and keeps the one the shipped `beats` judges better for the goal.
- **main.ts**: the `release` target. The project folder lives in the program's state directory.
