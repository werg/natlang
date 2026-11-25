# Review brief: judge one built program against the judged requirements

A reviewer reads one submission (`SUBMISSION/` with `program/`, `submission.json`, `NOTES.md`, `qualification/`) and
decides the judged requirements of `requirements.json`: **S4, S5, C3, D3, D5** (the program) and **N1** (the idea). The crisp
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
- **One canonical value per meaning.** If two values of a judgment's result type name the same thing on some item (a
  section that occurs once is both its `first` and its `last`), the type is wrong: S4 fails, naming the item and the
  two values.
- Every text written by people that a judgment reads (documents, notes, descriptions, messages, names, titles,
  nicknames, labels), including fields of records passed to it and text a TypeScript helper assembles from outside
  fields before passing it on, is typed `Untrusted<...>`. Identifiers, numbers and closed value sets need not be
  (`invented.py qualify` flags plain `string` fields as S6; you judge the rest).
- **The expected answer is the one a careful reader gives.** When you, reading an item carefully, reject the
  `expected` answer or a scripted reading, S4 is unmet: name the item and your reading, and mark S4 false even if the
  rest of the program is sound.
- **What the items say, the program can hold.** An item whose text moves state the types cannot express (a side
  statement, an aside, a second clause that says someone learned something another way) is a defect of the program,
  not of the item: S4 fails and the finding asks for the types and code that represent it.
- **Every rule a scripted answer rests on is written down**: in the instruction, a doc comment of an argument type, or
  code. A scripted answer that needs a precedence or resolution rule found nowhere in what the call sees fails S4; so
  does a doc-comment rule applied to some items and not others. `qualification/reference.txt` lists which children
  entry answered which item: read the items a shared entry answered with special care.
- Each judgment receives what it needs to answer and nothing it must not see (no expected answers, no other items'
  verdicts it should not depend on, no ids that leak the answer).

**S5. Reading for meaning.** No input note or field states a judgment's answer (a note "her surname is Le Guin" makes
the entry-element judgment a copy): name each such item. At least one judgment reads a passage of two or more
sentences. The `## Hardest items` of NOTES.md are items a careful reader could really get wrong, for the reason given.

**C3. Interaction.** At least one judgment reads the results of two earlier judgments, or state carried across items;
and at least two rules or stages interact so that their order or precedence decides the outcome. Name them.

**D3. Exact work in TypeScript.** Every exact step (arithmetic, counting, ordinals and positions, sorting, parsing,
record assembly, state updates, walks over items) is TypeScript. Judgments map words to referents: "the second right"
is read as `{side: 'right', ordinal: 2}` by the judgment and resolved to a road by code, or code numbers the roads
before the judgment sees them. `.nl` instructions state judgments of meaning and name the helpers they call; a root or
judgment that spells out a per-item walk, a count or an algorithm in prose fails D3.

**D5. An algorithm.** The exact logic includes at least one non-trivial algorithm the answer depends on (a graph walk, a
fixed point, a search, an enumeration), beyond sorting and grouping. Name it and the file.

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
    "S5": {"met": true, "findings": []},
    "C3": {"met": true, "findings": ["<the judgments and rules that interact>"]},
    "D3": {"met": true, "findings": []},
    "D5": {"met": true, "findings": ["<the algorithm and its file>"]},
    "N1": {"met": true, "findings": ["<nearest entries and the respects in which the idea differs>"]}
  },
  "F5": null
}
```

`findings` of an unmet requirement name each defect concretely (file, item id, the two readings, the untyped field), so
the builder can fix it without asking. `invented.py admit` admits only a submission whose review names its current
digest (any edit after the review needs a new review) and meets all six.
