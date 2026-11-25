# Build brief: turn one new idea into a program, its instances and an exact checker

The program inventor's `build` service hands this brief, the idea card (`idea.json`) and the requirements
(`requirements.json`) to a frontier coding agent working in an empty submission directory. The agent writes a
complete idiomatic natlang program for the idea, two or three input instances with exact checkers, and the controls
that qualify them. It never writes model trajectories, and it never writes a generator: every instance is literal data
written for this program.

Automated qualification admits a submission. There is no per-program review loop: a reviewer reads a random sample
of admitted programs afterwards, and what it finds changes this brief, not your program. So the quality of the
program is decided while you build it: the "One answer per item" section and the self-check are where that happens.

## Read first

1. `skills/natlang-authoring/SKILL.md`, then `references/composition.md` and `references/language.md` (the paradigm:
   types first, exact steps in TypeScript, each judgment one narrow typed function, `decide`, `@natlang/node/std`,
   higher-order judgments, `iterateOn(...).until(...)`, recursion with `measure:`, `ensures:`, `Is`/`Untrusted`,
   `Neuralese<T>`, folder transactions).
2. Two idiomatic programs: `examples/review/` and `applications/sem-merge/` (`merge.ts`, `intentsCompatible.nl`). They
   show the idioms; they are small on purpose. For the size and depth a submission has, read one pilot submission:
   `/home/werg/data/natlang-novel-programs/pilot-20261124/sub/nonvisual-interaction-parity-auditor/` (instances of
   1,700-2,000 words in 27-31 items, about 220 lines of exact TypeScript). Its idea is taken; its scale is the target.
3. `requirements.json` (the crisp requirements are checked by code; the judged ones are what a sampled reviewer
   grades).
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
                                the judgments in its own folder `<helper>/` and `@natlang/node/std`
      <helper>/...              a helper's own callable folder when it calls deeper judgments
  submission.json               the instances and controls (below)
  NOTES.md                      one paragraph per judgment: its trap and the evidence in the text that settles each
                                item; then `## Hardest items`: at least three items and why a careful reader could
                                get each wrong. NOTES explains; every rule an answer rests on is in the program.
  idea.json                     the idea card you built
```

SUBMISSION holds exactly these (qualification adds `qualification/`). Logs, the self-check, friction notes and
scratch scripts go beside it, in `SUBMISSION/..`.

The root `.nl` is what the student model executes in training: it receives the instance's inputs as its arguments,
works in a TypeScript eval scope where its callable folder's functions are bindings, and returns the typed answer.
Its instructions name the helpers it calls and the judgments of meaning it makes itself; exact work lives in the
TypeScript helpers. A root whose order of steps is fixed says it in one or two sentences naming each helper and what
it passes on.

## Depth

A submission is a program an expert would recognise as their own work, run on a real working session:

- **Instances at full size: one real working session.** Each instance is what the card's `inputs` describes: the
  day's calls, the whole rehearsal thread, the week's letters. Every item is there because the user would meet it and a
  judgment has to read it: a caller's directions run on for 100-150 words with a false start and a correction; a
  rehearsal thread has 20-30 messages, several about other songs. The card's prose size is binding (C2); its row and
  function counts are sketches, so reach the size with items that each need reading, never with rows that change
  nothing.
- **A judgment reads a passage.** At least one judgment reads two or more sentences and weighs them together (a whole
  transcript turn, a paragraph of a letter), and at least one reads the results of two earlier judgments, or state
  carried across items (the place a caller has reached, the key in force).
- **Words to referents; code counts.** A judgment reads what the words say, and TypeScript works out which item they
  pick. "Take the second right after the church" becomes `{ side: 'right', ordinal: 2, after: 'church' }` from the
  judgment, and code walks the map to the road. A choice among items known only at run time works the same way: code
  numbers or describes the candidates, and the judgment returns the description (or `decide`s one boolean per
  candidate, and code takes the most probable). Code also settles exact identity before it asks.
- **Text people wrote is `Untrusted`.** Messages, notes and descriptions, and also names, titles, nicknames and labels,
  including the fields of records a judgment receives. Identifiers, codes and dates say what they are in their name or
  doc comment (`roadId`, `/** ISO date */`); a closed set of values is a literal union.
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
of, restates it, disputes one of repairsSince, or leaves it unrelated. A letter that both narrows the complaint and
disputes a repair disputes the repair.
```

The code passes the complaint that an earlier judgment found open and the repair reports a second judgment matched to
it, so this call reads the results of two judgments; it then updates which complaints stand, in letter order.

## One answer per item

Every scripted answer must be the one a careful reader gives who sees only what the call sees: the instruction, the
doc comments of its argument types, and the item. These seven habits get you there.

1. **The item describes; the judgment names.** Write each item in its writer's own words about what happened, and
   leave the naming to the judgment. "We can do the 18th instead of the 12th now, if that still suits" is the text;
   `replaces` is the judgment's answer. A log line says what was measured ("p95 latency 6.2 s at 22:40"); a code
   comment says what the function is for ("Charge an invoice in full with a stored payment method"); a cleaning note
   says what the operator saw ("a glossy film that a cloth only spread around"). The answer comes from weighing that
   evidence. A value's own domain term is fine in the text (a step that adds DMSO, a judgment that names `dmso`).
2. **Judgments read values; code computes.** Every number, unit, date, duration and comparison is TypeScript. The
   judgment reads "about fifteen files" as `{ value: 15, approx: true }`, "6 lb per gallon of glucose" as
   `{ amount: 6, unit: 'lb/gal', substance: 'glucose' }`, "active from 37 °C to 50 °C" as
   `{ min: 37, max: 50, inclusive: true }`, "the march itself in three minutes" as `{ minutes: 3 }`; code holds the
   conversion table, adds the durations, applies the tolerance and decides whether the values agree. This holds inside
   pair judgments too: a judgment comparing two statements receives the code's comparison of their amounts and dates
   as an argument (`timing: 'different-days'`).
3. **One written rule, applied to every item.** When an answer needs a precedence or a resolution ("the surface the
   operator describes decides, ingredients only when the note describes none"), write it once in the instruction or a
   doc comment, or put it in code, and then derive each scripted answer by applying that rule to the item. Examples in
   an instruction are instances of its rule, item for item.
4. **Every definition has a direction and every overlap a winner.** A relation between two things says which side is
   which ("weaker: the earlier statement says less than the testimony"; "the base card's front defines the
   question"). When two written rules can both apply to one item, the instruction says which wins (the last sentence
   of the worked judgment above). An order rule is the order a reader of the items would follow (notes in the order
   they were sent).
5. **Types hold what the text can say.** An amount can be approximate (`approx: true`), a time can be a bound
   (`{ earliest: '2026-04-12' }` for "after the April 12 meeting", which can be the same afternoon), a statement can carry several
   facts (`facts: Fact[]`), a reading can be absent (`null`, with its doc comment saying when). Each value of a result
   type names one meaning: a section that occurs once in a song is the `only` one, or the doc comment says which value
   it takes.
6. **Expected answers come from the code.** The order in which code applies items, directives and conditions decides
   the result. Script the children, run `invented.py dry-run`, read every call it lists against its item, and take
   `expected` from the value it returns; when that value surprises you, read the code path that produced it before
   changing anything. A field you write beside the text (`celsius: 50`) is data you typed, once; a field a script
   computes from other fields is not (N2). A scratch script that lays out what you wrote (line numbers of a
   transcript, ISO dates from a calendar you typed) or validates it (applies your hand-written diffs) is fine.
7. **An ambiguous item is fixed by a rule or a type, and kept.** When an item has two defensible readings, add the
   rule that settles it (habit 3 or 4) or the type that represents it (habit 5). The hard item stays: it is the
   training signal.

## Self-check before you qualify

Write `SUBMISSION/../<slug>.SELFCHECK.md` with three short lists, one line per entry, and fix what they find. Run
`invented.py dry-run` first: its per-judgment listing is the list of items you check.

1. **Re-answer every scripted item from the program alone.** For each judgment, read its instruction and the doc
   comments of its argument types, close NOTES.md, and answer each item it is called on. Write: item, your answer,
   the script's answer. Every disagreement, and every item where you could defend two answers, is fixed with habits
   3-5.
2. **Every text a judgment sees, against that judgment's answer.** List each note, comment, title, label and doc
   comment that reaches a judgment call (the item's fields, the doc comments of its types, the source text a judgment
   reads), and whether it states or implies that judgment's answer. Reword each one that does (habit 1).
3. **Every number, date, unit and comparison.** List each one the answer depends on and the TypeScript line that
   computes or compares it. One a judgment computes moves to code (habit 2).

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
  opening (its instruction text, its arguments and the doc comments of the types it uses). Give each item its own
  entry, with one fragment from the judgment's instruction (within one line) and one unique to the item. An entry with
  the instruction fragment alone is a default for that judgment's remaining calls. After rewording an instruction,
  take its fragments again from the new text. A decision's value is the plain option (a record for a finite record
  result, `null` for a nullable one); a `Neuralese<string>` result is text. The value the eval returns must be accepted
  by the checker.

  The opening shows each argument as `name: Type = value`: a string as `hunkId: string = "r5019.1"`, an `Untrusted`
  string as a fenced data block holding the text verbatim (its header names the argument it came from), a record on
  one line up to its first text field, `event: Event = { eventId: "E15", ... }`. The safest item fragment is that
  line's start, `event: Event = { eventId: "E15"`: an id alone also matches calls that receive a list of the same
  records, and the name tells the two arguments of a pair judgment apart (`candidate: CardText = { cardId: "A01"`). A
  fragment may span lines (`"\nF\n```"` for a one-letter value). Keep instance ids out of `types.ts` doc comments, and
  make no id a substring of another.
- **ablations**: one per judgment. Its children are placed before the reference's, so they answer that judgment's
  calls with the trap reading; the ablated replay must return an answer the checker rejects (S2). A trap reading can
  make calls the reference never makes; answer those in the ablation's own children (a default entry is fine).
- **wrong_answers**: at least three per instance, each a named real mistake that qualify runs (F4): `children` for a
  misreading (placed before the reference's), `code` for an exact step done the wrong way (the root script with that
  step done otherwise), or both together (a reordering that makes new judgment calls). The replay computes the wrong
  answer; the checker must reject it. Add `value` only to check it against the replay. Take mistakes that pass the
  program's own checks: one the root's `ensures:` refuses never reaches the checker.
- **keyword_baseline**: the best rule over surface cues you can write without any `nl` call; it must run to an answer
  and be rejected (S3). It runs in the root's eval scope, where every callable-folder item is bound (a file's default
  export under the file's base name, a file of named exports as an object of them): give its locals other names, and
  name no folder file like a root argument. If your best keyword rule is accepted, the instance tests wording, not
  meaning: change it.
- **oracle**: `exact` (deep equality, record key order ignored) or `normalized` (also case, spacing and number format),
  or `verifier` with `"verifier": {"path": "training/semantic_fp_tasks/verifiers/<name>.mjs", "oracle": {...}}` when
  several answers are right.
- `semanticMap` and `semanticFilter` call their function with `(item, index)`: wrap a `.partial`-bound judgment in a
  lambda that passes only the item.

## Tools

Every command runs through the memory ledger on the DGX, at medium priority, not resumable, and waits for the job to
end (`--wait-exit`); start it as a background command and read the log when it completes. `LOGS` is an absolute
directory beside SUBMISSION; every run needs a new unit name (append -2, -3, ...):

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit pinv-<agent>-<what>-<n> --budget-gb 1.5 --reserve-gb 3 \
  --priority medium --no-resumable --wait-exit --wait 3600 --workdir /home/werg/natlang \
  -- /bin/bash -c '/home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/sources/invented.py <command> SUBMISSION > LOGS/<what>-<n>.log 2>&1'
```

- **`dry-run SUBMISSION [--controls]`** (seconds; `--controls` about half a minute): runs the program's TypeScript
  with your scripted answers and no model, and prints what the reference returns against `expected`, which children
  entry answered which call per judgment, the static checks, and with `--controls` what every wrong answer, ablation
  and keyword baseline returns. Iterate on this; it writes nothing into SUBMISSION.
- **`qualify SUBMISSION [--print-reference]`** (5-40 minutes): everything dry-run does, plus `natlang check` (F1), the
  dependence probes (C2), the stage cut and `SUBMISSION/qualification/report.json`. A run that is stopped leaves no
  report.json. Run it once dry-run is clean and the self-check is done; fix every problem it lists, never the checks.
- **`admit SUBMISSION --skip-qualify`**: admits the qualified submission (a stored report of this version under the
  current bar is used; anything else is qualified again).
- **`natlang check --json`** alone: the same ledger line with `--workdir SUBMISSION/program` and
  `/home/werg/.local/bin/natlang check --json > LOGS/check-<n>.json 2>&1`.

The bar is versioned: `report.json` records `bar` (its version and the digest of `requirements.json`) and the
submission's digest; the checks change only between waves.

What the crisp checks read:
- C2: the inputs hold at least the card's size in prose (the first "N words" of the card's `inputs`, over the whole
  instance; alphabetic words of texts of three words or more; ids, codes, timecodes and numbers do not count), and at
  least 800 words are in calls answered by an entry of their own that the result depends on: qualify gives each such
  entry the first different answer its judgment gives elsewhere in call order (or the other boolean), replays, and
  counts the entry when the returned value changes. "No answer depends on children entry N" is a warning, expected
  for readings that only route to another judgment; their text just does not count.
- Quotes: a fragment NOTES.md or a wrong answer's `mistake` quotes (three words or more, in double quotes) occurs in
  the instance or the program as they stand, compared as one string (a quote that wraps across a transcript's numbered
  lines does not match: quote within a line, or name the item by id).
- S6: a plain `string` a `.nl` function reads, as an argument or a record field, passes when its name says it is an
  identifier (`id`, `...Id`, `key`, `code`, `ref`, `url`, `path`, `date`, `time`, `...At`, `language`, `unit`) or its doc
  comment says so (`identifier`, `id`, `ISO`, `code`, `date`, `timestamp`, `timecode`, `URL`, `path`, `digits`,
  `computed`, in any case).
- S7: a relation word of an answer (`replaces`, `cancels`, `knows`) that every call holding it in its own text answers.
  Domain terms and words in a text several calls share are not flagged; prefer singular names for unit values.
- S8: a judgment asked about two identical values.
- C4: each `## Hardest items` entry names an item a judgment is called on, by an id that appears in that call (`m-a14`,
  `s22.2`) or by quoting a phrase of the call's text in double quotes.
- F3: the reference runs to an answer the checker accepts and takes at least half its children entries; an entry
  with an item fragment that answers several different calls is a warning (its fragment matches more than its item).

The model-backed idiom review (`natlang check --idiom --provider openrouter --model deepseek/deepseek-v4-flash`, after
`set -a; . /home/werg/.config/natlang/openrouter.env; set +a`) is optional; its findings are warnings.

Language facts builders met: callable-folder `.ts` is checked for loops like any program code (`for...of`, counted
loops, array methods, `iterateOn`); `Date`, `Intl`, `Map`, `Set` and `Math` are available there; text a helper builds
for an `Untrusted<string>` parameter that TypeScript checks (through `decide`, `.partial` or a function-typed slot)
goes through `untrusted(text, 'source')` from `@natlang/node`; types in an exported helper's signature are natlang
grammar types from `types.ts` (records, arrays, `Record<string, T>`, unions; no `Map`, no inline `Array<{...}>`).

## In a campaign

1. Take a card (one builder per card; exit 1 when the bank is exhausted), with the ledger line above running
   `PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python scripts/rl_tasks/program_atlas.py take BANK_DIR --by
   <agent> > LOGS/take-<agent>.json 2>&1`.
2. Build, dry-run, self-check, qualify until `report.json` is ok.
3. `invented.py admit SUBMISSION --skip-qualify` admits it: train ideas to the pool batch, dev and confirmation ideas to
   the held slice. A card you cannot build: `program_atlas.py release SLUG --status failed --reason "..."`.
4. Report. Admission ends your work on the program.
