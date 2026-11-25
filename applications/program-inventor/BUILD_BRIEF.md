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
2. Two idiomatic programs: `examples/review/` and `applications/sem-merge/` (`merge.ts`, `intentsCompatible.nl`). They
   show the idioms; they are small on purpose. For the size and depth a submission has, read one pilot submission:
   `/home/werg/data/natlang-novel-programs/pilot-20261124/sub/nonvisual-interaction-parity-auditor/` (instances of
   1,700-2,000 words in 27-31 items, about 220 lines of exact TypeScript). Its idea is taken; its scale is the target.
3. `requirements.json` (every requirement is checked; the crisp ones by code, the judged ones by a reviewer of another
   model family). Its numbers are floors the reviewer reads past: "Depth" below is what a submission looks like.
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
  NOTES.md                      one paragraph per judgment: its trap and the evidence in the text that settles each
                                item; then `## Hardest items`: at least three items and why a careful reader could
                                get each wrong
  idea.json                     the idea card you built
```

SUBMISSION holds exactly these (qualification adds `qualification/`, a reviewer `review.json`). Logs, friction notes
and scratch scripts go beside it, in `SUBMISSION/..`.

The root `.nl` is what the student model executes in training: it receives the instance's inputs as its arguments,
works in a TypeScript eval scope where its callable folder's functions are bindings, and returns the typed answer.
Its instructions name the helpers it calls and the judgments of meaning it makes itself; exact work lives in the
TypeScript helpers.

## Depth

A submission is a program an expert would recognise as their own work, run on a real working session:

- **Instances at full size.** Each instance is what the card's `inputs` describes, and at least 800 words of text
  people wrote or 15 items a judgment reads. Write the texts the way the real ones are: a caller's directions run on
  for 100-150 words with a false start and a correction; a rehearsal thread has 20-30 messages, several about other
  songs. Size comes from more and longer texts; the reference's model calls grow with it, and each case's episode
  budget (twice the reference's calls) grows along.
- **Evidence, not answers, in the data.** Each item carries the evidence a careful reader weighs: a donation slip says
  "née Klein, publishes under her married name", and the judgment works out the filing name from it. The NOTES
  paragraph names the evidence that settles each item.
- **A judgment reads a passage.** At least one judgment reads two or more sentences and weighs them together (a whole
  transcript turn, a paragraph of a letter), and at least one reads the results of two earlier judgments, or state
  carried across items (the place a caller has reached, the key in force).
- **Words to referents; code counts.** A judgment reads what the words say, and TypeScript works out which item they
  pick. "Take the second right after the church" becomes `{ side: 'right', ordinal: 2, after: 'church' }` from the
  judgment, and code walks the map to the road. A choice among items known only at run time works the same way: code
  numbers or describes the candidates, and the judgment returns the description (or `decide`s one boolean per
  candidate, and code takes the most probable).
- **Text people wrote is `Untrusted`.** Messages, notes and descriptions, and also names, titles, nicknames and labels,
  including the fields of records a judgment receives. Identifiers, codes and dates say what they are in their name or
  doc comment (`roadId`, `/** ISO date */`); a closed set of values is a literal union.
- **One value per meaning.** Each value of a result type names one meaning. A section that occurs once in a song is
  the `only` one; give the type that value, or state in its doc comment which value it takes.
- **An algorithm at the core.** About 100 lines of exact TypeScript, including one real algorithm the answer depends
  on: a walk over a graph, a fixed point, a search, an enumeration.

A worked judgment at this depth, from a tenancy-dispute program (the types in `types.ts`, the judgment in the folder,
the resolution in code):

```ts
/** One letter in the file, as received. */
export type Letter = { letterId: string; /** ISO date */ sent: string; from: Untrusted<string>; body: Untrusted<string> };
/** What a later letter does to one earlier complaint; `unrelated` when the letter does not take it up. */
export type Response = 'withdraws' | 'narrows' | 'restates' | 'disputes-repair' | 'unrelated';
```

```yaml
---
description: "What a tenant's later letter does to a complaint they made earlier."
args:
  complaint: Untrusted<string>
  repairsSince: Untrusted<string>[]
  letter: Letter
returns: Response
readout: decision
---
complaint is a complaint the tenant made, as worded then; repairsSince are the landlord's reports of work done since.
Read letter as a whole and decide what it does to complaint: withdraws it, narrows it to part of what was complained
of, restates it, disputes one of repairsSince, or leaves it unrelated.
```

The code passes the complaint that an earlier judgment found open and the repair reports a second judgment matched to
it, so this call reads the results of two judgments; it then updates which complaints stand, in letter order.

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
  opening (its instruction text, its arguments and the doc comments of the types it uses). Use one fragment from the
  judgment's instruction (within one line) and one unique to the item (an id in its arguments). Keep instance ids out
  of `types.ts` doc comments, and make no id a substring of another, or a fragment matches calls it was not meant for.
  An entry with only the instruction fragment answers every remaining call of that judgment: put the per-item entries
  first and such a default last. After rewording an instruction, take its fragments again from the new text. A decision's value is the plain option; a `Neuralese<string>`
  result is text. The value the eval returns must be accepted by the checker.
- **ablations**: one per judgment. Its children are placed before the reference's, so they answer that judgment's
  calls with the trap reading. Each ablated replay must return an answer the checker rejects (S2): this is what shows
  the answer depends on the judgment. A trap reading changes the state later steps see, so the ablated run can make
  calls the reference never makes (a location moved by the trap puts new pairs in reach of a pair judgment): walk the
  ablated run through and answer those calls in the ablation's own children. The replay names a call no entry answers.
- **wrong_answers**: at least three per instance, each the result of a named real mistake (a missed override, a wrong
  merge of two items, an exact step applied in the wrong order), never a random perturbation.
- **keyword_baseline**: the best rule over surface cues you can write without any `nl` call; it must run to an answer
  and be rejected (S3). It runs in the root's eval scope, where every callable-folder item is bound (a file's default
  export under the file's base name, a file of named exports as an object of them): give its locals other names. If your best keyword rule is accepted, the instance tests wording, not meaning: change it.
- **oracle**: `exact` (deep equality, record key order ignored) or `normalized` (also case, spacing and number format),
  or `verifier` with `"verifier": {"path": "training/semantic_fp_tasks/verifiers/<name>.mjs", "oracle": {...}}` when
  several answers are right.
- `expected` comes from running the program literally: each judgment read as the function would read it, receiving
  only its own arguments; each exact step as the code does it. Never the answer you intend.
- `semanticMap` and `semanticFilter` call their function with `(item, index)`: wrap a `.partial`-bound judgment in a
  lambda that passes only the item.

## Checks you run

Through the memory ledger on the DGX. `LOGS` is an absolute directory beside SUBMISSION (create it first):

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit pinv-qual-<slug>-<n> --budget-gb 1.5 --reserve-gb 3 --priority low --wait 3600 \
  --workdir /home/werg/natlang \
  -- /bin/bash -c '/home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py qualify SUBMISSION > LOGS/qualify-<slug>-<n>.log 2>&1'
```

`qualify` runs everything: `natlang check` (F1), packaging, the reference replay, wrong answers, ablations, keyword
baselines and the crisp requirements. A type check alone is the same command with `--workdir SUBMISSION/program` and
`/home/werg/.local/bin/natlang check --json > LOGS/check-<slug>-<n>.json 2>&1`. Both run in 1.5 GB, which the ledger
admits at once.

`run` waits in the foreground while the job is queued, so start it as a background command (or add `--detach`), and
read the log once `systemctl --user is-active pinv-qual-<slug>-<n>` is no longer `active`. Each run needs a new unit
name (append -2, -3, ...). It writes `SUBMISSION/qualification/report.json`; fix every problem it lists and rerun
until `ok` is true.

Qualification also replays each reference call by call: every case declares its episode budget (twice the reference
run's model calls) and each leaf judgment call becomes a stage task (`qualification/stages.jsonl`). Keep the idea card
you built as `SUBMISSION/idea.json`.

## In a campaign

1. Take a card: `PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python scripts/rl_tasks/program_atlas.py take
   BANK_DIR --by <agent-name>` (one builder per card; prints the card; exit 1 when the bank is exhausted).
2. Build and qualify as above until `report.json` is ok.
3. A reviewer (`REVIEW_BRIEF.md`) writes `SUBMISSION/review.json`; fix its findings, requalify, have it reviewed again.
4. `invented.py admit SUBMISSION` (through the memory ledger) admits it: train ideas to the pool batch, dev and
   confirmation ideas to the held slice. A card you cannot build: `program_atlas.py release SLUG --status failed
   --reason "..."`.
