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

`selectTrainingTasks.nl` is the separate typed RL scheduling policy. It consumes
train-task descriptors, scored outcome telemetry, resource allocation and
`TrainingCurriculumOptions`, and returns `{row,samples}[]`. The shared production
controller provides a crisp default through the same JSON interface and validates
row identities, group sizes and a rotating breadth reservation before collection.
The resolved options include `family_exploration_fraction` (default `0.5`):
half of practice explores unvisited tasks from least-covered semantic families.
Coverage uses distinct scored current-source task IDs; missing family metadata
remains one unknown bucket. Other practice preserves useful mixed outcomes while
preferring distinct families, and rotating breadth still advances its exact cursor.
An optional `select_tasks_command` can execute this natural-language function;
it receives one JSON request on stdin and returns the typed array on stdout.
The two advisory functions above retain their existing review-only behavior.
See `plans/neuralese/RL_CURRICULUM_20261016.md` for production bindings and
pending live measurement of newly authored task scaffolds.

Difficulty promotion uses `frontier_scope: "semantic-family"`: each declared family
advances from its own scored outcomes. Strong performance on math can therefore
move to harder math while tool execution continues practicing its own weak level.
The rotating breadth allocation and least-covered-family exploration stay active;
missing family metadata remains one unknown bucket.
