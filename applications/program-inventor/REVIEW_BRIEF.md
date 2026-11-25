# Review brief: one pass over a drawn sample of admitted authored programs

Reviews read a random sample of admitted programs (owner 2026-11-25: "reviews should only be on a drawn sample"; "it's
fine if individual samples are not perfect"). A review is one pass. It sends nothing back to the builder. Its main
product is what the next wave's brief should change; the wave report's statistics are the other half of that
feedback. Review as a model of another family than the builder when one is available, and never review a program you
built.

Draw the sample (admitted submissions without a review of their current version):

```
cd /home/werg/natlang
.venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py sample-reviews N [--seed S]
```

You only read and judge. Do not edit the submission. Write exactly one file, `SUBMISSION/review.json`.

## Read

1. `BUILD_BRIEF.md` (beside this brief): the program shape, the construct inventory and the trajectory format.
2. The submission: the root `.nl`, `types.ts`, the callable folders, `submission.json`.
3. `authored_programs.py check SUBMISSION` through the memory ledger (BUILD_BRIEF "Tools"): the call tree of each
   instance and its statistics.

## What to judge

1. **Grades, 1-5**:
   - `semantic_fp`: functions composed as values, combinators over semantic values, effects at the edges, crisp code
     only where exactness requires it.
   - `depth`: a deep tree of interior functions that decide and compose, not a flat list of leaves.
   - `synthesis`: NL literals built from data that the program then uses, and how much of the work runs through them.
   - `trajectory_realism`: whether each call's turns and answer read like what a competent model running that function
     would do and write.
   - `instance_realism`: a real working session at the card's size.
2. **`bad_answers`**: authored answers a careful reader of that call's own opening rejects (a summary that
   misstates its input, a decision the call's evidence contradicts), each with the instance, the call (function and
   item), and why. Training would teach these, so they are counted across the sample. Imperfect but defensible
   answers are not listed.
3. **`funnel_findings`**: up to five patterns this program shows that would recur in other builds, each as the change to
   `BUILD_BRIEF.md`, the wave's `EMPHASIS.md`, the authoring skill or the tooling that would prevent it, with one
   example from this program (file and call).

## `review.json` (`natlang.authored-review/1`)

```json
{
  "schema": "natlang.authored-review/1",
  "reviewer": "<your model id>",
  "digest": "<output of: .venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py digest SUBMISSION>",
  "grades": {"semantic_fp": 4, "depth": 3, "synthesis": 3, "trajectory_realism": 4, "instance_realism": 4},
  "bad_answers": [{"instance": "a", "call": "summarizeTopic (topic venue)", "authored": "<what it says>",
                   "why": "<what in the call's opening contradicts it>"}],
  "funnel_findings": ["<pattern> -> <change to BUILD_BRIEF/EMPHASIS/skill/tooling>; e.g. <file, call>"]
}
```
