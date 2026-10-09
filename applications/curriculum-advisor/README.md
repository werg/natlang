# curriculum-advisor: advisory judgments for collection

Plans/NATLANG_NATIVE_REVIEW.md, item P6. The rule ladders of `curriculum-policy.ts` are a data registry
(`ts-host/src/teacher/curriculum-rules.ts`); the two judgments below stay advisory.

```
judgeAnswerEquivalence.nl   a held extractive-answer row (qasper, musique): is the answer the same as the annotation?
nextCollectionBatch.nl      what the next collection batch should cover, from coverage against the target shares
index.ts                    crisp side: cards, coverage facts, checks, the release to human review
```

`judgeAnswerEquivalence` releases a row to a person and nothing else: `runtimeFailureReason` keeps holding it
(`unreviewed_extractive_answer_equivalence`), every release carries `admit: false` and `training_admission: "unchanged"`.
`nextCollectionBatch` proposes; nothing schedules a collection from it.

Run: `node ts-host/scripts/curriculum-advisory.mjs equivalence|next-batch ...` (see the header of that file). The output
files are labelled advisory with the advisor's hash, like the failure explainer's.
