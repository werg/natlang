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

- **Instances at full size: one real working session.** Each instance is what the card's `inputs` describes: the
  day's calls, the whole rehearsal thread, the week's letters. Every item is there because the user would meet it and a
  judgment has to read it: a caller's directions run on for 100-150 words with a false start and a correction; a
  rehearsal thread has 20-30 messages, several about other songs. The inputs hold the card's size in prose, and C2 counts
  the prose of the items a judgment reads with a children entry of their own whose answer the result depends on (at
  least 800 words), so size comes from items that each need reading and each matter.
- **The item describes; the judgment names.** The text says what happened in the words its writer would use, and the
  judgment's answer is the name for it. "We can do the 18th instead of the 12th now, if that still suits" is a new date
  for the same visit, and `supersedes` names it `replaces`; the message itself talks about dates, not about replacing.
  A value's own domain term is fine in the text (a step that adds DMSO, a judgment that names the additive `dmso`); the
  text holds the evidence, and the reader still has to weigh it. qualify reports a relation word that settles a
  judgment alone (S7: every call whose text says "replaces" answers `replaces`).
- **Every rule a scripted answer rests on is written down.** Each scripted answer follows from the judgment's
  instruction, its argument types' doc comments and the item, read literally. When the answer needs a precedence or
  resolution rule (an estimate leaves a standing policy in force; emphasis words are not claims), write the rule into
  the instruction or the doc comment, or put it in code, and apply it to every item the same way.
- **The program covers what the items show.** When an item describes a case the types cannot hold (a retcon that says
  the character heard it another way), extend the types and the code to represent it, and keep the item.
- **A judgment reads a passage.** At least one judgment reads two or more sentences and weighs them together (a whole
  transcript turn, a paragraph of a letter), and at least one reads the results of two earlier judgments, or state
  carried across items (the place a caller has reached, the key in force).
- **Judgments read values; code compares.** A judgment reads limits, conditions and amounts into typed values, and
  TypeScript compares them: the datasheet's "active from 37 °C to 50 °C" becomes `{ min: 37, max: 50, inclusive: true }`,
  the step's "incubate at 50 °C" becomes `{ celsius: 50 }`, and code decides whether 50 lies in the range. Code also
  settles exact identity before it asks: two identical texts are the same utterance without a judgment.
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
    "wrong_answers": [{"mistake": "<the real mistake>", "children": [{"match": ["..."], "value": "<the misreading>"}]},
                      {"mistake": "<an exact step done the wrong way>", "code": "<the root script with that step wrong>"}],
    "keyword_baseline": "<root eval script with no nl call that answers from surface cues; ends with return>"
  }]
}
```

- **reference** runs the program for real: the root eval calls the program's own helpers, the TypeScript runs, and
  each judgment call is answered by the first `children` entry whose `match` fragments all appear in the call's
  opening (its instruction text, its arguments and the doc comments of the types it uses). Use one fragment from the
  judgment's instruction (within one line) and one unique to the item (an id in its arguments). Keep instance ids out
  of `types.ts` doc comments, and make no id a substring of another, or a fragment matches calls it was not meant for.
  Give each item its own entry: `qualify --print-reference` lists, per judgment, which entry answered which item, so an
  entry that answers several items is visible, and each such item gets the same answer for the same reason. An entry
  no call of the reference takes is listed as a warning (it may serve an ablation); a reference that takes fewer than
  half its entries fails F3. After rewording an instruction, take its fragments again from the new text.
  A decision's value is the plain option (a record for a finite record result); a `Neuralese<string>` result is text.
  The value the eval returns must be accepted by the checker.

  The opening a fragment is found in shows each argument as `name: Type = value`: a string as `note: string =
  "the text"`, an `Untrusted` string as a quoted data block holding the text verbatim, a record as
  `step: Step = { stepId: "RT 4", celsius: 50, text: ... }`. Good fragments are an id with its key (`stepId: "RT 4"`)
  or a short phrase of the item's own text, within one line.
- **ablations**: one per judgment. Its children are placed before the reference's, so they answer that judgment's
  calls with the trap reading. Each ablated replay must return an answer the checker rejects (S2): this is what shows
  the answer depends on the judgment. A trap reading changes the state later steps see, so the ablated run can make
  calls the reference never makes (a location moved by the trap puts new pairs in reach of a pair judgment): walk the
  ablated run through and answer those calls in the ablation's own children. The replay names a call no entry answers.
- **wrong_answers**: at least three per instance, each a named real mistake that qualify runs (F4): `children` for a
  misreading (entries placed before the reference's, like an ablation's) or `code` for an exact step done the wrong way
  (the root script that runs the helpers with that step done otherwise). The replay computes the wrong answer, so it
  is always the answer that mistake gives; the checker must reject it. Add `value` only to check it against the replay.
  Take mistakes that pass the program's own checks: one the root's `ensures:` refuses never reaches the checker, so it
  is no wrong answer.
- **keyword_baseline**: the best rule over surface cues you can write without any `nl` call; it must run to an answer
  and be rejected (S3). It runs in the root's eval scope, where every callable-folder item is bound (a file's default
  export under the file's base name, a file of named exports as an object of them): give its locals other names. It
  has no scripted children, so it uses only the helpers that make no judgment call. If your best keyword rule is accepted, the instance tests wording, not meaning: change it.
- **oracle**: `exact` (deep equality, record key order ignored) or `normalized` (also case, spacing and number format),
  or `verifier` with `"verifier": {"path": "training/semantic_fp_tasks/verifiers/<name>.mjs", "oracle": {...}}` when
  several answers are right.
- `expected` comes from running the program literally: each judgment read as the function would read it, receiving
  only its own arguments; each exact step as the code does it. Script the children, run `qualify --print-reference`,
  read every call it lists against its item, and take `expected` from the value it prints.
- **Inputs are written, once.** Each input fact is written by hand in one place. A structured field the code reads
  (`celsius: 50` beside the step text) is fine when you wrote it; a field computed by a script from other fields is
  not data (N2). A scratch script beside SUBMISSION may lay out what you wrote as JSON.
- `semanticMap` and `semanticFilter` call their function with `(item, index)`: wrap a `.partial`-bound judgment in a
  lambda that passes only the item.

## Checks you run

Through the memory ledger on the DGX. `LOGS` is an absolute directory beside SUBMISSION (create it first):

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit pinv-qual-<slug>-<n> --budget-gb 1.5 --reserve-gb 3 --priority low --wait 3600 \
  --workdir /home/werg/natlang \
  -- /bin/bash -c '/home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py qualify SUBMISSION > LOGS/qualify-<slug>-<n>.log 2>&1'
```

The bar is versioned: `report.json` records `bar` (its version and the digest of `requirements.json`) and the
submission's digest, and the checks change only between waves, so the bar you start against is the bar you finish
against. `admit` qualifies again unless the stored report is of this submission under the current bar.

`qualify` runs everything: `natlang check` (F1, after rebuilding a stale runtime), packaging, the reference replay,
wrong answers, ablations, keyword baselines and the crisp requirements. `--print-reference` also prints, per instance,
the value the reference returns, the prose words its judgment calls read, and for each judgment which children entry
answered which item (the same text is in `qualification/reference.txt`). What the crisp checks read:
- C2: the inputs hold at least the card's size in prose (alphabetic words of texts of three words or more; ids, codes,
  timecodes and numbers do not count), and at least 800 words are in calls answered by an entry of their own that the
  result depends on: qualify gives each such entry another answer of its judgment, replays, and counts the entry when
  the returned value changes. An entry no answer depends on is a warning, and its text does not count.
- Quotes: a fragment NOTES.md or a wrong answer's `mistake` quotes (three words or more, in double quotes) occurs in
  the instance or the program as they stand (C4, F4).
- S6: a plain `string` a `.nl` function reads, as an argument or a record field, passes when its name says it is an
  identifier (`id`, `...Id`, `key`, `code`, `ref`, `url`, `path`, `date`, `time`, `...At`, `language`, `unit`) or its doc
  comment says so (`identifier`, `id`, `ISO`, `code`, `date`, `timestamp`, `timecode`, `URL`, `path`, `digits`, `computed`).
- S7: a relation word of an answer (`replaces`, `cancels`, `knows`) that every call holding it in its own text answers.
  A value naming a substance or term the text names (`dmso`, `household`) is never flagged, nor a word in a text
  several calls share (a datasheet).
- S8: a judgment asked about two identical values.
- C4: each `## Hardest items` entry names an item a judgment is called on by an id that appears in that call (`m-a14`,
  `s22.2`), an item whose text leaves the reading to the reader.
- F3: the reference runs to an answer the checker accepts and takes at least half its children entries. A type check alone is the same command with `--workdir SUBMISSION/program` and
`/home/werg/.local/bin/natlang check --json > LOGS/check-<slug>-<n>.json 2>&1`. Both run in 1.5 GB, which the ledger
admits at once.

The model-backed idiom review (`natlang check --idiom`) runs on a hosted model; the default profile's local server is
not running. Run it the same way, with `--workdir SUBMISSION/program`:

```
-- /bin/bash -c 'set -a; . /home/werg/.config/natlang/openrouter.env; set +a; /home/werg/.local/bin/natlang check --json --idiom --provider openrouter --model deepseek/deepseek-v4-flash > LOGS/idiom-<slug>-<n>.json 2>&1'
```

Its findings are warnings: weigh each against the skill.

`run` waits in the foreground while the job is queued, so start it as a background command (or add `--detach`), and
read the log once `systemctl --user is-active pinv-qual-<slug>-<n>` is no longer `active`. Each run needs a new unit
name (append -2, -3, ...). It writes `SUBMISSION/qualification/report.json`; fix every problem it lists and rerun
until `ok` is true.

Qualification also replays each reference call by call: every case declares its episode budget (twice the reference
run's model calls) and each leaf judgment call becomes a stage task (`qualification/stages.jsonl`). Keep the idea card
you built as `SUBMISSION/idea.json`.

## In a campaign

1. Take a card (one builder per card; exit 1 when the bank is exhausted). A ledger unit's output goes to the
   journal, so write it to a file and read that:

   ```
   python3 /home/werg/natlang/scripts/memory_ledger.py run --unit pinv-take-<agent> --budget-gb 1.5 --reserve-gb 3 --priority low --wait 3600 \
     --workdir /home/werg/natlang \
     -- /bin/bash -c 'PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python scripts/rl_tasks/program_atlas.py take BANK_DIR --by <agent> > LOGS/take-<agent>.json 2>&1'
   ```
2. Build and qualify as above until `report.json` is ok.
3. A reviewer (`REVIEW_BRIEF.md`) writes `SUBMISSION/review.json`; fix its findings, requalify, have it reviewed again.
4. `invented.py admit SUBMISSION` (through the memory ledger) admits it: train ideas to the pool batch, dev and
   confirmation ideas to the held slice. A card you cannot build: `program_atlas.py release SLUG --status failed
   --reason "..."`.
