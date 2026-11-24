# Build brief: turn one new idea into a program, its instances and an exact checker

The program inventor's `build` service hands this brief, the idea card (`idea.json`) and the requirements
(`requirements.json`) to a frontier coding agent working in an empty submission directory. The agent writes a
complete idiomatic natlang program for the idea, two or three input instances with exact checkers, and the controls
that qualify them. It never writes model trajectories, and it never writes a generator: every instance is literal data
written for this program.

## Read first

1. `skills/natlang-authoring/SKILL.md`, then `references/composition.md` and `references/language.md` (the paradigm:
   types first, exact steps in TypeScript, each judgment one narrow typed function, `decide`, `@natlang/node/std`,
   higher-order judgments, `iterateOn(...).until(...)`, recursion with `measure:`, `ensures:`, `Is`/`Untrusted`,
   `Neuralese<T>`, folder transactions).
2. Two idiomatic programs: `examples/review/` and `applications/sem-merge/` (`merge.ts`, `intentsCompatible.nl`).
3. `requirements.json` (every requirement is checked; the crisp ones by code, the judged ones by a reviewer of another
   model family).
4. If the checker is a verifier: the header comment of each `training/semantic_fp_tasks/verifiers/*.mjs` is its
   oracle schema.

## What to write

```
SUBMISSION/
  program/                      the natlang package the task runs
    natlang.json                {"schema": "natlang.package/v2", "name": "@invented/<slug>", "version": "0.1.0"}
    package.json                {"name": "@invented/<slug>", "private": true, "type": "module"}
    types.ts                    every record and union; doc comments on fields say formats, units, what null means
    <root>.nl                   the root: the operation a user calls, returning a named type from types.ts
    <root>/                     the root's callable folder
      <judgment>.nl             one narrow judgment each (`readout: decision` for finite answers)
      <helper>.ts               exact TypeScript: parsing, ordering, aggregation, state, graph walks; it may import
                                sibling `.nl` judgments and `@natlang/node/std`
      <helper>/...              a helper's own callable folder when it calls deeper judgments
  submission.json               the instances and controls (below)
  NOTES.md                      one paragraph per judgment: its trap and why each instance item has one reading
```

The root `.nl` is what the student model executes in training: it receives the instance's inputs as its arguments,
works in a TypeScript eval scope where its callable folder's functions are bindings, and returns the typed answer.
Its instructions name the helpers it calls and the judgments of meaning it makes itself; exact work lives in the
TypeScript helpers.

## `submission.json` (`natlang.invented-program/1`)

```json
{
  "schema": "natlang.invented-program/1",
  "slug": "<idea slug>",
  "author": {"inventor": "<model id>", "builder": "<model id>"},
  "root": "<root>.nl",
  "constructs": ["decide", "std-collection", "higher-order", "..."],
  "judgments": [{"name": "<function or inline name>", "file": "<root>/<judgment>.nl or the .ts holding the inline nl", "trap": "<the wrong reading>"}],
  "instances": [{
    "name": "a",
    "level": 3,
    "inputs": {"<root arg>": "<value>"},
    "oracle": "exact",
    "expected": "<the one correct answer>",
    "alias_answers": [],
    "verifier": null,
    "reference": {
      "root": [["eval", {"code": "const r = await <helper>(...); return r;", "finish": true}]],
      "children": [{"match": ["<fragment of the judgment's instruction>", "<fragment unique to this item>"], "value": "<correct answer>"}]
    },
    "ablations": [{"judgment": "<name>", "children": [{"match": ["..."], "value": "<the trap reading>"}]}],
    "wrong_answers": [{"value": "<wrong answer>", "mistake": "<the real mistake that produces it>"}],
    "keyword_baseline": "<root eval script with no nl call that answers from surface cues; ends with return>"
  }]
}
```

- **reference** runs the program for real: the root eval calls the program's own helpers, the TypeScript runs, and
  each judgment call is answered by the first `children` entry whose `match` fragments all appear in the call's
  opening (its instruction text and its arguments). Use one fragment from the judgment's instruction (within one line)
  and one unique to the item (an id in its arguments). A decision's value is the plain option; a `Neuralese<string>`
  result is text. The value the eval returns must be accepted by the checker.
- **ablations**: one per judgment. Its children are placed before the reference's, so they answer that judgment's
  calls with the trap reading. Each ablated replay must return an answer the checker rejects (S2): this is what shows
  the answer depends on the judgment.
- **wrong_answers**: at least three per instance, each the result of a named real mistake (a missed override, a wrong
  merge of two items, an exact step applied in the wrong order), never a random perturbation.
- **keyword_baseline**: the best rule over surface cues you can write without any `nl` call; it must run to an answer
  and be rejected (S3). If your best keyword rule is accepted, the instance tests wording, not meaning: change it.
- **oracle**: `exact` (deep equality, record key order ignored) or `normalized` (also case, spacing and number format),
  or `verifier` with `"verifier": {"path": "training/semantic_fp_tasks/verifiers/<name>.mjs", "oracle": {...}}` when
  several answers are right.
- `expected` comes from running the program literally: each judgment read as the function would read it, receiving
  only its own arguments; each exact step as the code does it. Never the answer you intend.

## Checks you run

From `SUBMISSION/program/`, through the memory ledger on the DGX:

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit pinv-check-<slug>-<n> --budget-gb 3 --wait 3600 \
  --workdir $PWD \
  -- /bin/bash -c '/home/werg/.local/bin/natlang check --json > ../../logs/check-<slug>.json 2>&1'
```

Then the full qualification (packaging, reference replay, wrong answers, ablations, keyword baselines, the crisp
requirements):

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit pinv-qual-<slug>-<n> --budget-gb 4 --wait 3600 \
  --workdir /home/werg/natlang \
  -- /bin/bash -c '/home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py qualify SUBMISSION > SUBMISSION/../qualify-<slug>.log 2>&1'
```

The ledger queues a job until memory frees up (minutes on a busy DGX); `run` returns once the job's unit has started,
so wait until the unit has ended (`systemctl --user is-active pinv-qual-<slug>-<n>` is no longer `active`) and then
read the output file. Each run needs a unit name not used before (append -2, -3, ...). It writes
`SUBMISSION/qualification/report.json`. Fix every problem it lists and rerun until `ok` is true.
