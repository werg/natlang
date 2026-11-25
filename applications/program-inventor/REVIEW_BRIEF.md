# Review brief: judge one built program against the judged requirements

A reviewer reads one submission (`SUBMISSION/` with `program/`, `submission.json`, `NOTES.md`, `qualification/`) and
decides the judged requirements of `requirements.json`: **S4, C3, D3** (the program) and **N1** (the idea). The crisp
requirements are already decided by `invented.py qualify`: do not redo them, but read `qualification/report.json`
(metrics, reference calls, stage counts) as context. Review as a model of another family than the builder when one is
available, and never review a program you built.

You only read and judge. Do not edit the submission. Write exactly one file, `SUBMISSION/review.json`.

## Read first

1. `requirements.json` (beside this brief): the statement of each requirement is what you hold the program to.
2. `skills/natlang-authoring/SKILL.md` and `references/language.md` (what `Untrusted<T>`, `readout: decision`,
   `ensures:`, std combinators and `.partial` mean).
3. The submission: the root `.nl`, `types.ts`, every file of the root's callable folder, `submission.json` (both or all
   three instances, each instance's reference children, ablations, wrong answers), `NOTES.md`.

## What to check

**S4. Narrow judgments, one defensible answer, Untrusted outside text, the right data.** For each judgment `.nl` (and
inline `nl`):
- It asks one question of meaning with a typed answer; not a bundle of questions, not exact work.
- **One defensible answer on every item of every instance.** Read each item the judgment is called on (the reference
  children tell you which items and which answer the author scripted) and answer it yourself, literally, as the
  function receiving only its own arguments would. A scripted answer you would not give, or an item where a careful
  reader could defend two answers, fails S4: name the item and both readings. The pilot's typical defect: notes worded
  as associations ("heavy users are far less likely to cancel") where the judgment asks whether an influence is
  stated.
- Every free text from outside (documents, notes, descriptions, messages, titles, labels written by people) that a
  judgment reads is typed `Untrusted<...>`, including text a TypeScript helper assembles from outside fields before
  passing it on. Identifiers, numbers and closed labels need not be.
- Each judgment receives what it needs to answer and nothing it must not see (no expected answers, no other items'
  verdicts it should not depend on, no ids that leak the answer).

**C3. Interaction.** At least one judgment reads the result of another judgment, or of an exact stage computed from
judgments; and at least two rules or stages interact so that their order or precedence decides the outcome. Name them.

**D3. Exact work in TypeScript.** Every exact step (arithmetic, counting, sorting, parsing, record assembly, state
updates, walks over items) is TypeScript. `.nl` instructions state judgments of meaning and name the helpers they call;
a root or judgment that spells out a per-item walk or an algorithm in prose fails D3.

**N1. Novelty.** Build the current atlas and retrieve the nearest existing ideas:

```
cd /home/werg/natlang
PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python scripts/rl_tasks/program_atlas.py atlas -o /tmp/<you>-atlas.md
```

Read the atlas groups "Ideas this process already accepted or is building" and the entries nearest the idea
(`SUBMISSION/idea.json`, or the bank card the builder took). Compare the idea with each nearest entry in four respects:
who runs it and for what repeated job; the judgments of meaning at its core; the shape of its computation (how exact
steps and judgments feed each other, what state is carried, where results combine); the kind of input data. A change of
subject matter alone counts as no difference. N1 is met when the idea differs from every entry in at least two
respects; it fails when some entry matches it in three or four. Also say whether the program as built still is the idea
(a build that shrank the idea into a different, more common one fails N1).

**Optional: F5 adjudication.** When the task names a live run whose answer the checker rejected, run the program's
instructions literally on that instance at each difference and rule `expected-wrong`, `answer-wrong` or `ambiguous`
(`adjudicate.nl`). Put it under `"F5"` with the differences and the culprit calls.

## `review.json` (`natlang.invented-review/1`)

```json
{
  "schema": "natlang.invented-review/1",
  "reviewer": "<your model id>",
  "digest": "<output of: .venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py digest SUBMISSION>",
  "requirements": {
    "S4": {"met": true, "findings": []},
    "C3": {"met": true, "findings": ["<the judgments and rules that interact>"]},
    "D3": {"met": true, "findings": []},
    "N1": {"met": true, "findings": ["<nearest entries and the respects in which the idea differs>"]}
  },
  "F5": null
}
```

`findings` of an unmet requirement name each defect concretely (file, item id, the two readings, the untyped field), so
the builder can fix it without asking. `invented.py admit` admits only a submission whose review names its current
digest (any edit after the review needs a new review) and meets all four.
