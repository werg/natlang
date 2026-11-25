# Build brief: a semantic functional program and its authored execution trajectories

You build one natlang program from an idea card, two or three literal input instances, and for each instance the
whole execution trajectory: every model call the program makes, with what it receives and what it answers. The
trajectories are training data: each model call becomes one training row for a small model (LFM3-1.7B) learning to
run natlang programs. Nothing runs your program with a model. A replay plays your trajectory through the real runtime,
and the only check is that it is consistent with the program. The reasoning between actions is filled in later by
another stage. Never write it yourself.

The owner's direction (2026-11-25):

> "Just have the author write the entire execution trajectory to train on and then all we need to do is collect the
> reasoning in between the actions."

> "It absolutely has to be a deep code base of purely natural language functions able to call each other. It should
> very much, in the then generated trajectory, use a lot of the NatLang specific functionality."

> "Remember we want to do semantic functional programming."

> "we actually have the true ability to do function synthesis at runtime, because all this is semantic. And based on
> the data, we can create new functions." ... "It's just built in our ability to make these nl literals."

> "it's fine if individual samples are not perfect"

Statistics over every wave's trajectories steer the next wave (`EMPHASIS.md`). No review loop follows admission. A
reviewer reads a drawn sample afterwards, and what it finds changes this brief, not your program.

## Read first

1. The wave's `EMPHASIS.md`: what the previous wave's statistics say to do more of.
2. `skills/natlang-authoring/SKILL.md`, then `references/composition.md` (the paradigm) and `references/language.md`
   (front matter, callable folders, eval).
3. `spec/SPEC.md`: "Natural-language functions" (inline `nl` literals), "Captures", "Iteration and termination",
   "Postconditions", "Model surface" (decision readout), "Eval", "Types" (function types, `.partial`).
4. Examples that show the constructs: `examples/screening/` (a `.nl` function that returns a judgment, `.partial`),
   `examples/outline/` (structural recursion with `ensures:`), `examples/release/` (`semanticRepair`, `fixedPoint`,
   folder transactions), `examples/refunds/` (`.in(context)`, `decide`, a folder transaction).
5. The worked mechanics example, small on purpose:
   `/home/werg/data/natlang-novel-programs/authored-examples/thread-digest/` (`program/` and `submission.json`). It
   shows the trajectory forms below (a reduce over messages, an interior function's own turns, a decision with
   `decide`, a `.nl` function that synthesizes a judgment, NL literals built from data, `iterateOn` with an NL stopping
   predicate, an `ensures:` predicate). Your program is several times its size and depth.

## The program: semantic functional programming

A program is a **deep tree of typed natural-language functions calling each other**: pure functions from typed
arguments to a typed result, composed like functions in a functional language, over immutable data, with effects only
at the edges.

- **Deep.** About 10-20 `.nl` functions. The root calls functions that have their own callable folders, and those call
  functions with folders of their own: four or more levels from the root to the deepest leaf. An interior function
  receives a whole item (a letter, a scene, a chapter), decides which of its helpers the item needs, calls them, and
  composes what they return. Leaves answer one question about one thing.
- **Composed as values.** Functions are passed, bound, returned and mapped: an `.nl` argument typed as a function,
  `fn.partial(value)`, `fn.in(NatlangContext.empty().with({ policy }))`, a `.nl` function whose result type is a
  function.
- **Functions synthesized at run time.** An NL literal is a function value made in eval code. When its instructions
  are composed from data the program has derived, the program creates a new function from its data: a criterion
  for each category it found, a rubric an earlier call wrote, the house style it read from examples, a test written
  for this one case. It is then called, passed, mapped and composed like any other function. Every program does this.
- **Combinators over semantic values.** Map, filter, reduce, match and rank over items; `iterateOn` until a check
  holds; recursion on a smaller argument; repair and fixed point; folder reducers and transactions for programs that
  read and write documents.
- **Open-ended outputs.** Most functions write something: a summary, a reply, a rewritten passage, a plan, an
  explanation, an extracted record with text fields. Synthesized functions write too: a writer built from the voice
  of the source, a reply drafted for this customer's history, not only tests that return a boolean. A decision (`readout: decision`) is right where the honest
  answer is one of a few values. The root returns what the user actually needs, which is usually a document with
  structure, not a single label.
- **Crisp TypeScript only where exactness or the outside world requires it.** By default a program has no `.ts` file
  besides `types.ts`. The few lines of exact work a function needs (filtering by a field, grouping, counting,
  assembling the result record) go in that function's own eval code, in its trajectory. A callable-folder `.ts`
  helper is for exact work that is long or reused (a date calculation, a parser of a fixed format, a graph walk over
  ids); list each in `submission.json` `crisp` with the reason. Never write an algorithm out in prose: a step that is
  exact is code, in eval or in a helper.
- **Types first.** `types.ts` holds the records and unions with doc comments on fields (formats, units, what `null`
  means). Text people wrote is `Untrusted<string>`. A property you would otherwise ask for in a sentence is
  `Is<T, "...">`.

### The idea card

Take the card's user, world and input size. Its `judgments` and `program_sketch` are a starting point. Its
`crispWork`, `output ... compared exactly`, `whyKeywordsFail` and `constructs` fields were written for an earlier bar
that made programs into TypeScript algorithms with narrow judgments. Redesign the program to the shape above: the
exact search or walk the card describes becomes a composition of natural-language functions with a few lines of eval
code, and the output becomes what the user would want to read.

## The construct inventory: each construct in source and in the trajectory

The replay records every call. Your trajectory answers each call below the root with an entry in `trajectory.calls`
(next section). The table shows what each construct looks like in the source and what you author for it.

| Construct | Source | In the trajectory |
|---|---|---|
| Named function | `assess.nl` in the caller's folder | a call of `assess`: an entry `{fn: "assess", match: [item fragment], value}` |
| Interior function | `summarize.nl` with folder `summarize/` | `{fn: "summarize", match: [...], turns: [["eval", {code}], ["return_result", {...}]]}`: its eval calls its own items |
| Decision | `readout: decision` on a finite result | `value` is the chosen option; in a caller, `decide(fn, ...args)` returns `{ value, probabilities, confidence }`: branch on `.confidence` where a doubtful answer should go another way (a second judge, `escalate`, a rule) |
| Pick one of several | one decision over the candidates (`returns:` a union of their ids, or `semanticRank`) | one call that sees every candidate, not a filter of N booleans of which one is true |
| NL literal | ``nl<(x: T) => R>`instructions`(arg)`` in eval | a call named `nl@eval:N`, matched by a fragment of its instructions |
| NL literal built from data | ``nl<(g: Gist) => boolean>`g bears on ${topic}` `` in eval, `topic` derived by the program | its creation in the eval code, then its calls, matched by the interpolated text |
| Function synthesized by a function | `returns: "(item: Item) => boolean"` in front matter | that call's turn is `["eval", {"code": "return nl<(item: Item) => boolean>`...${...}...`;", "finish": true}]`; the returned function's calls nest under it. Its literal is all those calls see: carry into it what the program derived about this case (this character's usual look, the rubric an earlier call wrote) and every clause of the synthesizing function's instructions, not only a name |
| Function as argument | `args: { judge: "(item: Item) => boolean" }` | the receiving call's eval calls `judge(item)`; `check` prints the calls of `judge` under the receiving call |
| Partial application | `meets.partial(rule)` in eval | the bound function is passed or mapped; its calls carry the bound argument |
| Rebinding | a leaf: `verdict.in(NatlangContext.empty().with({ policy }))`; a function with its own callable folder: `withContext(examine, { policy })`, which keeps its companion items | calls of the rebound function see `policy` in their opening |
| Map | `await semanticMap(items, assess)` | a `[semanticMap]` span, one call per item under it |
| Filter | `await semanticFilter(items, test)` (a function, or a criterion string judged by the shipped `meets`) | a `[semanticFilter]` span, one call per item; a string criterion asks `meets` (`criterion: string`, `item: unknown`) |
| Partition | `await semanticPartition(items, classify)` → `{ groups, uncertain }` | a span, one call per item |
| Reduce, carrying state | `await semanticReduce(thread, [], (gists, m) => readMessage(m, gists).then(g => [...gists, g]))` | a span, one call per item, each seeing the state so far |
| Match | `semanticMatch(left, right, { relation })` (two lists: old and new versions, claims and sources, requirements and findings) | one `relates` call per candidate pair: `{fn: "relates", match: ["left: unknown = { id: \"r2\"", "right: unknown = { id: \"f7\""], value: true}`; with `match: fn` the pair calls are calls of `fn` |
| Rank | `semanticRank(items, goal)` | debiased by default: `beats` in both orders of every pair, n(n-1) calls (`{ debiased: false }`: n(n-1)/2); entries `{fn: "beats", match: ["first: unknown = { id: \"x4\"", "second: unknown = { id: \"x2\""], value}` |
| Repair | `semanticRepair(draft, { check, fix, attempts })` | the `fix` calls, one per failed check; a check that holds on the first draft is a correct trajectory with no fix call |
| Fixed point | `fixedPoint(revise, first)` (a value revised until a revision only rewords it: a summary tightened, a plan reconciled) | each step's calls, then `settled` on `revision: Revision = { previous: ..., current: ... }`: `false` while the step still changes what the value says, `true` when it only rewords |
| `iterateOn` with an NL check | ``await iterateOn(draft => revise(draft, notes), first).until(nl<(d: Draft) => boolean>`...`)`` | the check is called on the first state and after every step (`false`, step, ..., `true`); each step's calls. Each check's opening shows its state, and earlier states may be quoted in it: match a sentence only this state contains. A check still false after about ten steps also asks a `progress_judge` call |
| `iterateOn` with a measure | `.withMeasure(state => state.pending.length).until(state => state.pending.length === 0)` | each step's calls until the code check holds |
| Structural recursion | `recursive: structural` (or `measure:`), the function calls itself on a part: a reply and its replies, a section and its subsections, a claim and the claims it depends on | one call per level, each with its own entry. Types may be recursive, results included (`type Reply = { text: Untrusted<string>; replies: Reply[] }`). A graph walk recurses on the nodes not yet visited (a shorter array), not on a growing visited path. Data that relies on itself in a cycle (claims supporting each other) is a counted loop of passes or a `fixedPoint` |
| Postcondition | `ensures:` with `check:` (code, no call) or `predicate:` with `against: [args]` | a `predicate` is judged by a call named `holds` with `value: unknown`, `predicate: string` and `context: unknown` (the `against` values): `{fn: "holds", match: ["value: unknown = <start of the value>"], value: true}` |
| Refinement | `refine(value, 'predicate', { context })` in eval, or a result typed `Is<T, "...">` | one `holds` call per value produced (`value: unknown`, `predicate: string`). A refinement that fails: `holds` answers `false` for the first value, and the producing call's entry has `turns` with two `return_result` turns, the refused value and the corrected one (the refusal is not an `expected_failures` action) |
| Folder reducer | root `kind: directory-reducer` with `writes:`; the instance gives `folder_files` (a program that maintains a folder: a manuscript's chapter files, a suite's test files, a household's notes) | the root's turns use `list_files`, `read_file`, `write_file`, `edit_file`; `folder.apply(reducer, ...)` runs a nested reducer |
| `pluggable` | `pluggable({ crisp, nl }, setting)` in a helper | a `pluggable` span and the side it ran |
| Escalation, context | `escalate(fn, { to, floor })`, `withContext(fn, { policy })` | the wrapped function's calls |
| Long value (learned compaction) | nothing to write: an argument or an eval output over 8,000 characters (a whole document, a long log) | the runtime cuts it and calls the learned operator `view` on the whole value: `{fn: "view", match: ["answers: <start of the receiving function's instructions>"], value: "<the summary>"}`. Its opening is `instructions:` (for an argument `The call that receives this argument as doc (string) answers: ...`; for an eval output `Read this eval result for: ...`) and `value:` (the whole value). Write what that purpose needs from the value: names, numbers, where things are. The receiving call's listing shows your summary, as the operator's Neuralese state, in the value's place (`doc: string = <summary>  // view of the value; doc holds all of it`); an eval output shows its cut, then `Compacted value: ` and the state. Its answer may use what the summary says; eval code still reads the whole variable |
| Neuralese value | `returns: Neuralese<string>` (a gist, a running state handed on); a consumer takes `gist: Neuralese<string>` | the producing call's `value` is the text; each consumer's opening shows that text where the value stands. Pass it on whole; `await read(g)` in eval is a model call with its own entry |

Front matter is YAML: a `description:` containing `: ` goes in double quotes. Eval code follows the iteration rules: `for...of`, counted loops, array methods, `iterateOn`; no `while`. Outside text
is never interpolated into an NL literal's instructions: pass it as an argument (`untrusted-instruction`), and build
instructions from values the program derived (a category name, a rubric a call wrote, a field's id).

## The instances

Two or three instances, each one real working session of the card's user: the whole thread, the day's calls, the
week's letters, as people write them. Every call in the trajectory is yours to write, so size an instance by the items
you can read and answer with care: typically 10-25 items and 800-2,000 words of prose. The card's size is an upper
sketch. Every input is literal data written for this program: no generator, no template, no variants.

Make the instances differ in structure, not only in names: which of the card's traps occur, in what order, how much of
the input is beside the point (a real chat is mostly not about money). Each trap of the card's `instance_sketch` should
occur in some instance and reach the root's output. A union type that reads the source's own vocabulary covers every
value the source uses (screenplay headings say `DAY` and `CONTINUOUS`), or its doc comment says how each maps.

## The trajectory

**Write every answer yourself, reading that call's opening as the model would.** A script may assemble the JSON from
answers you wrote (and lay out data you typed), but no script computes an answer. If a script could compute a call's
answer from the call's arguments (a turn is "left" because its angle is negative, an item matches because two ids are
equal), that step is exact: it belongs in eval code, and the natural-language call should ask something only reading
can answer.

For each instance you write the whole execution: what the root does, turn by turn, and what every call below it
answers, including the open text a function writes (the summary, the reply, the rewritten passage). Write what a
careful, competent model running the program would do and write. Answers need not be perfect. They must be
consistent with the program: each value has the declared type, each eval runs, each call your code makes has an
entry.

- **An answer uses what its call's opening shows.** You know the whole instance; the call knows its instructions, its
  arguments and its captures. A writer that receives a ledger entry cannot mention the birthday in the chat or the
  exchange rate in the rules. When the answer needs a fact, change the program to pass it (add the field, the note, the
  rate as an argument). `check` lists open answers that state numbers or names their opening does not show.
- **Loops get honest work.** A draft a careful model writes follows its own instructions. Never break a rule a call's
  instructions state so that a repair, an `iterateOn` or a refinement has something to do: training would teach the
  broken draft. A check that holds on the first state is a correct trajectory with zero steps. Loops and refinements
  get real work from what a careful first answer honestly misses: a property the producing call was not told or could
  not see (an `ensures:` predicate judged `against` the source the writer only summarized, a promise the policy in the
  check's context does not allow), a revision that still changes the meaning (`fixedPoint`), a judgment that needs a
  second look. Answer that `holds` `false` and author the corrected answer.
- **Refinements on final values.** Put `Is<T, "...">` on the value a function finally returns, not on drafts a later
  repair fixes: a draft typed as the refinement is judged before the repair can run.
- **A literal's instructions refer to its parameters and interpolated values.** Anything else it needs is a parameter
  (bind it with `.partial`) or interpolated text the program derived.

```json
{
  "schema": "natlang.authored-program/1",
  "slug": "<idea slug>",
  "author": {"builder": "<model id>"},
  "root": "<root>.nl",
  "crisp": [{"file": "<root>/<helper>.ts", "why": "<the exactness or outside-world reason>"}],
  "instances": [{
    "name": "a",
    "inputs": {"<root arg>": "<value>"},
    "trajectory": {
      "root": [["eval", {"code": "..."}], ["eval", {"code": "...; return result;", "finish": true}]],
      "calls": [
        {"fn": "readMessage", "match": ["message: Message = { messageId: \"m3\""], "value": {"...": "..."}},
        {"fn": "summarizeTopic", "match": ["topic: string = \"venue\""],
         "turns": [["eval", {"code": "const verdict = await decide(isSettled, topic, gists);\nverdict.value"}],
                   ["return_result", {"status": "success", "value": {"...": "..."}}]]},
        {"match": ["states a constraint that any decision on venue", "messageId: \"m6\""], "value": true}
      ]
    }
  }]
}
```

- **`root`**: the root call's turns in order. An eval's local `const`s persist to the next eval, with object identity:
  `items.indexOf(x)` and `===` work across evals. The last turn is
  `["eval", {..., "finish": true}]` returning the result, or `["return_result", {"status": "success", "value": ...}]`.
  Look at a result before using it the way a model would (end an eval with the value to inspect), and split work into
  several evals where a model would want to see an intermediate result.
- **`calls`**: one entry per call below the root. `fn` names the function. `match` fragments must all appear in the
  call's opening: its instructions and its arguments as shown, `name: Type = value` (`topic: string = "venue"`), a
  record on one line starting `message: Message = { messageId: "m3"`. Take the start of the argument's line as the
  item fragment: an id alone also matches every call whose arguments include that record (the `earlier` list of a
  reduce, for one). The entry with the most fragments that match answers the call. An entry with only `fn` answers
  every remaining call of that function, which is right for an interior function whose turns are the same code
  (eval code reads the call's own parameters). Give every entry `fn` (`holds`, `beats`, `relates`, `meets`,
  `settled` and `nl@eval:N` included): an entry without it can answer another function's call whose arguments quote the
  same text. A string copied from `Untrusted` text shows as a fenced block (```` ```untrusted data from argument chat of
  settleTrip ````, then the text, then ```` ``` ````), inside records too: end a fragment before the fence
  (`message: Message = { id: "m4"`) or take the whole block. A fragment taken from an earlier record or an earlier
  draft also matches later calls that quote it (a reduce's `earlier` list, a revision that keeps the draft's first
  sentence): take a passage only this call shows. Two calls with identical openings are answered by one entry. An NL
  literal's opening also declares the variables it captured: every
  visible variable its instructions name, as an ordinary word too ("could not be played" captures a `played` array), and
  every variable interpolated into it. A fragment can match there; the check names the variable when it does. Reword
  the instructions or list the captures with `nl.with({ … })` when a word captures a variable the call does not need.
- **`value`** answers in one turn (a decision's option, or the result). **`turns`** plays several turns:
  `[["eval", {"code": ...}], ["return_result", {"status": "success", "value": ...}]]`, or a final eval with
  `"finish": true`.
- **`expected_failures`** (optional, per instance): when the trajectory deliberately includes actions that fail and are
  then corrected (a type refused and fixed), their number.
- **`rl`** (optional, per instance): `{"oracle": "exact" | "normalized" | "verifier", "expected": ...}` when the final
  answer has one checkable value; the instance then also becomes an RL task. Most instances leave it out.
- **`folder_files`** (optional, per instance): the folder a directory-reducer root works on, `{"path": "text"}`.

## Tools

Every command runs through the memory ledger on the DGX, at medium priority, not resumable, waiting for the job to end.
Start it as a background command and read the log when it completes. `LOGS` is an absolute directory beside
SUBMISSION; each run needs a new unit name (append -2, -3, ...):

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit <unit>-<what>-<n> --budget-gb 2 --reserve-gb 3 \
  --priority medium --no-resumable --wait-exit --wait 3600 --workdir /home/werg/natlang \
  -- /bin/bash -c 'PYTHONPATH=training/neuralese:. .venv-neuralese/bin/python scripts/rl_tasks/sources/authored_programs.py <command> SUBMISSION > LOGS/<what>-<n>.log 2>&1'
```

- **`check SUBMISSION`** (well under a minute): `natlang check` on the program, then the replay of every instance's
  trajectory. First it lists every call with no entry, in every stage at once: `NO ENTRY ...` with the call's whole
  opening (instructions and arguments) and a suggested entry (`fn` and the first argument line no other call of that
  function shows). A missing call is answered by a placeholder of its type so the replay reaches the later stages;
  `ok` stays false until every call has an entry. Then each instance's call tree (each call, its nesting and what it
  returned; calls through a function argument under the call that made them), the returned value in full, every failed
  action in full, the instance's statistics, and warnings: an entry that answered nothing, one value that answered
  calls with different arguments, an entry that answered calls of several functions, a call that several entries
  match with different answers, open answers stating what their call does not show. `ok: true` means the trajectory
  is consistent. Iterate on this. `--no-natlang-check` skips the compile step for a quicker replay.
  Neuralese is replayed as the serving engines read it: every learned-compaction site makes its `view` call (a
  missing one is `NO ENTRY ... for a call of view`, and its suggested fragment is the line of its instructions), and
  `Neuralese<T>` values are read as their text in place. The statistics count `operator_calls`.
- **`dump SUBMISSION`**: the whole replay as JSON in `SUBMISSION.dump.json` beside SUBMISSION (`--out FILE` elsewhere):
  every call's opening and the entry that answered it, the call tree with arguments, eval code and values in full,
  failed actions, the statistics. Read it instead of writing a dump wrapper.
- **`admit SUBMISSION`**: checks again and admits: the trajectories go to the training pipeline, the idea to the
  campaign's ledger.
- **`stats SUBMISSION`**: the statistics alone (JSON).

The std combinators are built into every call's eval scope by name (no import). They pass a `.nl` function only the
arguments it declares, so `semanticMap(items, assess)` works with a one-parameter `assess`.

## Statistics, not checks

`check` prints the statistics the wave report aggregates. They steer the campaign and are never a gate:

- the share of calls by construct, and the construct kinds the trajectories use;
- the depth of the NL call tree and the number of interior functions;
- NL instruction lines against TypeScript logic lines, and NL calls against helper calls;
- higher-order uses: functions passed, bound, returned, iterated;
- NL literals: how many are created, how many are built from data, the calls made through them, and the share of the
  call tree that runs through them;
- answer lengths, and the mix of open, structured, selection (text copied from the arguments) and closed outputs;
- the shipped judgments and refinements by answer (`holds` false counts the refinements that failed and were repaired);
- open answers stating numbers or names their call's opening does not show.

Aim at the targets the output names. Where your program falls short of one, consider the design again rather than
padding: a construct used because it fits the work is the training signal; one bolted on is noise.

## In a campaign

1. Take a card: the wave brief gives the `program_atlas.py take` line, which writes the card as `SUBMISSION/idea.json`.
   One builder per card.
2. Design the types and the function tree, then write the program. Then write instance a and its trajectory, run
   `check` until `ok: true`, then the other instances.
3. `admit` admits it. A card you cannot build: `program_atlas.py release SLUG --status failed --reason "..."`.
4. Report. Admission ends your work on the program.
