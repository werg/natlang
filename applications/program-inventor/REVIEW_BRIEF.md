# Review brief: one sampled pass over an admitted program

Automated qualification admits invented programs; reviews read a random sample of them (owner 2026-11-25: "reviews
should only be on a drawn sample"). A review is one pass with three outputs: grades, the expected answers a careful
reader rejects, and the patterns this program shows that the build brief and the authoring skill should change. It
sends nothing back to the builder: there are no requests and no rounds. Review as a model of another family than the
builder when one is available, and never review a program you built.

Draw the sample (admitted submissions without a review of their current version):

```
cd /home/werg/natlang
.venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py sample-reviews N [--seed S]
```

You only read and judge. Do not edit the submission. Write exactly one file, `SUBMISSION/review.json`.

## Read

1. `requirements.json` (beside this brief), and the "One answer per item" section of `BUILD_BRIEF.md`: the habits
   the program was built to.
2. `skills/natlang-authoring/SKILL.md` and `references/language.md` as needed.
3. The submission: the root `.nl`, `types.ts`, every file of the root's callable folder, `submission.json`, `NOTES.md`,
   `qualification/report.json` and `qualification/reference.txt` (which children entry answered which item). Run
   `invented.py dry-run SUBMISSION` through the memory ledger (`--priority medium --no-resumable --wait-exit`, as in
   BUILD_BRIEF's Tools) when you want the calls listed fresh.

## What to do

1. **Recompute each instance's expected answer as a careful reader would**, from the items and the program's written
   rules (instructions, doc comments, code), not from the builder's simulations. Re-answer the scripted judgment calls
   along the way, the shared entries and the NOTES.md hardest items first.
2. **`wrong_expected`**: list each place where the expected answer, or a scripted reading it rests on, is one a careful
   reader rejects: the instance, the item and judgment, what the script says, what the careful reader says, and why.
   Only that. An item where two readings are both defensible, a rule written nowhere, a judgment doing arithmetic, a
   note that states its answer: these are funnel findings, not wrong expected answers. A non-empty `wrong_expected` on
   the admitted version keeps the program out of every batch published from now on; it is not sent back for a fix.
3. **Grades, 1-5**: idiomatic natlang, semantic depth, code-based depth, instance realism and size, novelty (for
   novelty, the atlas: `PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python scripts/rl_tasks/program_atlas.py
   atlas -o /tmp/<you>-atlas.md`, the entries nearest the idea compared on user and job, core judgments, computational
   shape and input kind). Mark each judged requirement (S4, S5, C3, D3, D5, N1) met or not, in one line each.
4. **`funnel_findings`**: up to five patterns this program shows that would recur in other builds, each as the change
   to `BUILD_BRIEF.md`, the authoring skill or the tooling that would have prevented it, with one example from this
   program (file and item). These are the review's main product: they are read across the sample and folded into the
   brief and skill.

## `review.json` (`natlang.invented-review/1`)

```json
{
  "schema": "natlang.invented-review/1",
  "reviewer": "<your model id>",
  "digest": "<output of: .venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py digest SUBMISSION>",
  "grades": {"idiomatic_natlang": 4, "semantic_depth": 3, "code_based_depth": 4, "instance_realism_and_size": 4, "novelty": 4},
  "requirements": {"S4": {"met": true, "findings": ["<one line>"]}, "S5": {"met": true, "findings": []},
                   "C3": {"met": true, "findings": []}, "D3": {"met": true, "findings": []},
                   "D5": {"met": true, "findings": []}, "N1": {"met": true, "findings": []}},
  "wrong_expected": [{"instance": "b", "item": "u34 (contradicts, entry 109)", "script": "consistent",
                      "careful_reader": "contradicts", "why": "<the text that settles it>"}],
  "funnel_findings": ["<pattern> -> <change to BUILD_BRIEF/skill/tooling>; e.g. <file, item>"]
}
```

A review names the digest of the version it read; `invented.py publish` reads `wrong_expected` only from a review of
the admitted version.
