# RL task authoring kit (2026-11-03)

For model workers who author new RL tasks for the LFM3 student. Owner decision (2026-11-03): fill the missing target
behaviours mainly with on-policy RL, so the student needs many genuinely different natlang programs with exact
oracles. You write programs, inputs, exact answers and controls. You never write model trajectories.

Read this whole file before writing anything. Rules that override everything else:

- **Run the program literally, then write `expected`.** Simulate every helper as a reader that sees only the
  arguments it receives (and its own text), in the order the program calls them. Never copy a dataset's gold: if the
  program's routing, budget, filter or rule drops a fact, the literal answer changes and `expected` follows the
  program. Round 1 found no arithmetic slips in hand-computed answers; every wrong `expected` came from answering by
  intent or from dataset gold instead of running the program as written.
- **Every fact the answer needs is visible in the inputs or in the text of the helper that decides it.** No hidden
  gold, and no rule in a root preface the deciding helper never receives. If a careful human with the inputs could not
  get exactly the expected value, the task is wrong.
- **Every natural-language judgment has one clearly right reading.** If a note does not name the item, a careful
  reader must not be able to argue the other way. Prefer notes that name the box, barcode or entity. When ambiguity is
  the lesson, the program's own stated rule must settle it. Ambiguous tasks are rejected, not repaired later.
- **Drop a tag rather than simulate it.** A construct you cannot build honestly is not claimed: `constructs: []` is a
  valid task. Do not write your own inbox to claim `folder_reducer` (use `folders`), do not tag
  `multi_operation_codebase` on a one-operation world, do not give a skill a fact the answer does not need.
- **The answer contract is visible.** The root's `returns` type and its instruction say exactly what form the answer
  takes (record fields, units, whole numbers, exact names), and the question asks for exactly what the root returns
  (a sum is not "the average", a count is not "which").
- **Real diversity, not renaming and not toggling.** A new task must differ in what is computed, its call graph, its
  state, or its failure modes. Swapping names, numbers or cities, switching a rule sentence on or off, or adding an inert
  paragraph is the *same* task (see section 7). The novelty gate and the audit both remove padding and count it
  against the batch.
- **Never use protected or held data** (see Exclusions). Never copy text from evaluation packets.
- **Policy is natural language; exact work is code.** NL helpers interpret, judge and summarize; arithmetic, sums,
  sorting, counting, comparisons and state updates happen in the root's code (`eval`) or are spelled out as exact steps.
- **Do not spell out a wrong form in an instruction** ("do not write X"). Say what to do.

## 1. Where to write

Staging root: `/mnt/external/natlang-rl-authoring/`.

```
/mnt/external/natlang-rl-authoring/
  runtime/                     frozen verifier runtime (do not edit)
  examples/tasks.jsonl         the three examples below, already qualified
  batches/<worker>/<batch>/    YOUR work: tasks.jsonl, then qualified/ and novelty.json
  admitted/                    supervisor-only: batches that passed every gate
  rotation.jsonl               rotation ledger (section 7)
```

One batch = 10-40 tasks from one rotation cell, in `batches/<worker>/<batch>/tasks.jsonl` (one JSON object per line).
Use `<batch>` = `b0001`, `b0002`, ... per worker.

## 2. Task file format (`natlang.rl-authored-task/1`)

One JSON object per line:

| field | meaning |
|---|---|
| `schema` | `"natlang.rl-authored-task/1"` |
| `id` | lowercase letters, digits, dashes; unique for you, e.g. `"pantry-close-day-0007"` |
| `author` | `{"model": "<your model id>", "worker": "<your worker name>"}` |
| `family` | snake_case name of the *design theme* (the reusable application/program family), e.g. `stock_day_reconciliation`. All tasks that run the same program design share one family, and the novelty gate needs every family in a batch to gain 5+ genuinely different programs (or already have 20+): put your 6+ different programs of one theme under ONE family name (e.g. `contract_obligation_trackers`), never one family per program. Aim for 20+ genuinely different programs per family over time. |
| `shape` | short name of the call-graph pattern, e.g. `carried-notes-then-exact-apply` |
| `domain` | free word: `operations`, `planning`, `finance`, `science`, `customer_service`, `code`, ... |
| `world` | identity of the scenario world. Tasks that share facts (same store, same document set, same codebase, same customer) MUST share `world`; the split (train/dev/confirmation) is derived from it, so related tasks never cross splits. |
| `level` | honest difficulty 0-3: 0 = one helper call and a direct answer (a two-call relay or a one-call filter is level 0, not 2); 1 = two or three steps; 2 = several dependent steps or carried state; 3 = long inputs, many items, corrections or competing constraints. Level 3 must be genuinely long and constrained: a dozen or more items, corrections that interact, or rules that compete and need a stated precedence. Round 1 labelled one-call filters 2-3 and had no real level-3 task; auditors relabel inflated levels. Every family needs real level-3 tasks. For constructs that need at least two helpers (relay, fan-in, depth 3), level 0 is the simplest two-call version: short clean inputs, no corrections, each helper used once. |
| `constructs` | list, possibly empty, drawn from: `call_depth_3`, `neuralese_relay`, `typed_neuralese`, `neuralese_fan_in`, `iterate`, `iterate_neuralese`, `folder_reducer`, `decision_readout`, `skills`, `multi_operation_codebase`, `effect_state` (definitions in section 3). Claim only what the program really does; `[]` is correct for a good task whose construct would be fake. |
| `root` | the root file name, e.g. `"close_stock_day.nl"` (top level, ends in `.nl`) |
| `files` | object path -> file text. The root plus its helpers. Helpers callable by the root live in the folder named like the root without `.nl` (`close_stock_day/absorb.nl`). A helper called by a helper lives in that helper's folder (`handle_claim/assess_parts/coverage_window.nl`). Skills: `<folder of the function that reads them>/skills/<skill-name>/SKILL.md`: the runtime binds a folder's skills only to the call that owns that folder (the root owns `close_stock_day/`, so `close_stock_day/skills/x/SKILL.md` is visible to the root and to nobody below it; a helper `handle/assess.nl` needs `handle/assess/skills/x/SKILL.md`). Qualification rejects a skill read by any other function. |
| `inputs` | object: the root's arguments by name, with concrete values |
| `expected` | the exact correct result, matching the root's `returns` type |
| `oracle` | `"exact"` (deep equality; records compare by field, key order irrelevant) or `"normalized"` (also ignores case, surrounding spaces and number formatting for strings/numbers) |
| `oracle_depth` | `"effect-state"` when `expected` is the full resulting state of the world after the task's operations (a stock ledger, a schedule, a set of bookings, a repaired config); otherwise `"answer"`. Must agree with the `effect_state` construct. |
| `wrong_answers` | 2-6 *meaningful, task-specific* wrong results of the right type: what a model gets by a real mistake (missed correction, wrong branch, off-by-one, dropped item, applied rule in the wrong order). Each must differ from `expected` and must be rejected by the oracle. |
| `alias_answers` | 0-3 alternative forms of the SAME correct result. They become the oracle's `alternates`, so the runtime accepts them; key order is already ignored and `normalized` already ignores case, spacing and number formatting. An alias may not equal a wrong answer. Use them only for forms a careful author would also call right. |
| `smoke` | `{"code": "<root eval script>", "children": {"<helper name>": <scripted answer>}}`. The code calls the helpers the way the program intends (one pass is enough); each helper answers with its scripted value. It proves the call graph wires and type-checks. Typed Neuralese helpers answer with a record; `Neuralese<string>` helpers with text; `readout: decision` helpers with one of their options. |
| `groups` | optional list of extra world/source identities (e.g. the dataset item id your task was inspired by) |
| `reference` | optional; default is a direct return of `expected`. Either a list of root actions, or `{"root": [...], "children": [{"match": ["<text the helper call receives>"], "value": <its answer>}, ...]}` when the root's script calls helpers (required for folder tasks: the reference must really do the moves/writes). |
| `folders` | required with `folder_reducer` (qualification rejects the tag without it): object relative path -> file text, e.g. `{"inbox/r-0412.md": "..."}`. These files are the task's world folder; the root (which must have `kind: directory-reducer`) reads, moves and writes them. Never have the root create its own items or take them as arguments instead. The folder is carried through conversion, reference replay and smoke. |
| `expected_folders` | optional, with `folders` and `oracle_depth: "effect-state"`: the complete folder after a correct run (every file that must exist, with its exact text). Files left behind or misplaced fail the task. |
| `wrong_code` | optional (needs `reference`): 1-4 wrong root scripts that replace the reference's last eval, e.g. a sort that skips one category or moves everything to one place. Each must be rejected. Use these for effect tasks: a wrong returned value alone does not show the oracle checks the effects. |

For the smoke of a per-item helper (one decision per file), give the child answer as
`{"$match": [{"match": "<text in its call>", "value": <answer>}, ...]}` instead of one value.

### `.nl` file format

```
---
args:
  name: "TypeScript type as a string"
returns: "TypeScript type as a string"
---
Instruction text: numbered steps over named data. Name helpers with their call, e.g.
`const notes = await absorb(notes, message);`.
```

Optional frontmatter line `readout: decision` (before `args:`) makes a helper a decision readout: its `returns` must be a
finite union of string literals (or `boolean`), and the parent calls `await decide(helper, ...args)`, getting
`{ value, ... }`. Types: `string`, `number`, `boolean`, arrays `T[]`, records `{ a: string; b: number }`,
`Record<string, number>`, string-literal unions, `Neuralese<string>`, `Neuralese<{ ...record... }>`. Use optional
fields or `null`, never `undefined`.

## 3. Constructs (what each tag requires)

| tag | the program must contain |
|---|---|
| `call_depth_3` | root -> helper -> sub-helper, each solving a *different* subproblem. The middle helper owns its own work (it combines its sub-helper's result with something else it computes or decides) and actually calls the sub-helper. A pass-through middle, a sub-helper that is a one-line comparison (exact comparisons belong in code), or the same task delegated again is not depth. Qualification rejects a missing chain and warns on pass-through and thin middles (heuristic). |
| `neuralese_relay` | a helper returns `Neuralese<...>` and that value is passed as an argument into another helper |
| `typed_neuralese` | a helper returns `Neuralese<{ ...record... }>` |
| `neuralese_fan_in` | two or more Neuralese-producing helpers (distinct roles, or one per item) whose outputs the parent combines |
| `iterate` | the root uses `iterateOn` over several items/steps (not one) and states its stopping condition in words ("Stop when ..."); qualification rejects any `iterateOn` without one |
| `iterate_neuralese` | `iterateOn` whose carried state is (or contains) a Neuralese value |
| `folder_reducer` | the root has frontmatter `kind: directory-reducer` and works over the task's `folders` items (read each file with `folder.files("dir/*.md")` and `file.readText()`, act with `file.moveTo("dir/")` or writes); usually paired with `effect_state` + `expected_folders`. See the example below. |
| `decision_readout` | a `readout: decision` helper whose distribution the parent acts on via `decide(...)`. A readout is a leaf: it runs in one pass with no tools, so it calls no helper and owns no callable folder (qualification rejects both); do its lookups in the parent first and pass the facts in. |
| `skills` | `<folder>/skills/<name>/SKILL.md` files, in the folder of the function that reads them, that the program loads (`read_code("skills.<name>")`) and applies to choose or perform work. A skill must hold the only statement of a fact or procedure the answer needs: a skill that restates the note is decoration, and the audit removes the tag. |
| `multi_operation_codebase` | the task's root is one of several operations of the same reusable program: tasks of one `world` run different root functions over the same helper files. A world whose tagged tasks run one root fails qualification. |
| `effect_state` | `expected` is the complete resulting state, checked field by field |

Small `folder_reducer` example (`returns_desk_sorting`, level 1; qualified: reference, 4 negatives rejected, alias
accepted, smoke; generator `/mnt/external/natlang-rl-authoring/examples/folder/make_folder_example.py`):

```json
{"root": "sort_returns.nl", "constructs": ["folder_reducer", "decision_readout", "effect_state"],
 "oracle_depth": "effect-state", "inputs": {},
 "files": {"sort_returns.nl": "---\nkind: directory-reducer\nargs: {}\nreturns: \"{ restock: number; refurbish: number; scrap: number }\"\n---\nSort the returned items waiting in returns/.\n1. For each file in returns/, read its note and let assess decide what happens to the item.\n2. Move the file to <decision>/<same name> (restock/, refurbish/ or scrap/). Leave nothing in returns/.\n3. Return how many files went to each decision.\n",
           "sort_returns/assess.nl": "---\nreadout: decision\nargs:\n  note: \"string\"\nreturns: \"\\\"restock\\\" | \\\"refurbish\\\" | \\\"scrap\\\"\"\n---\nDecide what to do with a returned item from its inspection note: ...\n"},
 "folders": {"returns/r-0412.md": "Kettle, box unopened, customer changed their mind. Seal intact.\n",
             "returns/r-0414.md": "Toaster with a burnt-out heating element and a melted cord. Smells of smoke.\n"},
 "expected_folders": {"restock/r-0412.md": "Kettle, box unopened, customer changed their mind. Seal intact.\n",
                      "scrap/r-0414.md": "Toaster with a burnt-out heating element and a melted cord. Smells of smoke.\n"},
 "expected": {"restock": 1, "refurbish": 0, "scrap": 1},
 "reference": {"root": [["eval", {"code": "const counts = { restock: 0, refurbish: 0, scrap: 0 };\nfor (const file of await folder.files(\"returns/*.md\")) {\n  const d = await decide(assess, (await file.readText()).trim());\n  await file.moveTo(d.value + \"/\");\n  counts[d.value] += 1;\n}\nreturn counts;", "finish": true}]],
               "children": [{"match": ["Kettle, box unopened"], "value": "restock"}, {"match": ["Toaster with a burnt-out"], "value": "scrap"}]},
 "wrong_code": ["<the same script, but scrap files are not moved>", "<the same script, but every file goes to restock/>"],
 "smoke": {"code": "<the reference script>", "children": {"assess": {"$match": [{"match": "Kettle", "value": "restock"}, {"match": "Toaster", "value": "scrap"}]}}}}
```

Negative and alias answers of a script reference replace only its final `return`, so the moves still happen and the
returned value is what is judged.

## 4. Qualify your batch (mandatory)

Run on the DGX through the memory ledger (one command; it converts, replays, smoke-runs, checks custody and writes a
report):

```sh
cd /home/werg/natlang
B=/mnt/external/natlang-rl-authoring/batches/<worker>/<batch>; U=rlauth-<worker>-<batch>
rm -f $B/qualify.exit
python3 scripts/memory_ledger.py run --unit $U --budget-gb 4 --priority low --wait 3600 -- \
  sh -c "env PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang \
  /home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/qualify_authored.py $B/tasks.jsonl $B/qualified \
  > $B/qualify.log 2>&1; echo \$? > $B/qualify.exit"
# `run --wait` only waits for ADMISSION: it returns when the unit starts (or is queued), and its exit code says
# nothing about the job. Wait for the job itself and check how it ended:
until [ -f $B/qualify.exit ] || { ! systemctl --user is-active -q $U && ! systemctl --user is-active -q $U-waiter; }; do sleep 5; done
sleep 2; echo "job exit: $(cat $B/qualify.exit 2>/dev/null || echo MISSING)  unit result: $(systemctl --user show $U -p Result --value)"
cat $B/qualified/report.json | head -60
```

The batch is qualified only when `job exit: 0` AND `report.json` says `"result": "PASS"`. `job exit: MISSING` means
the job was killed or never ran (unit result `oom-kill`, `signal`, `exit-code` ...): read `$B/qualify.log` and rerun;
never read an old `report.json` as the result of this run. (A unit that finished successfully is removed by systemd,
so its result then shows `success`; a failed unit stays listed with its failure.)

What it checks per task:
1. **convert**: the JSON matches section 2 (errors name the line and field);
2. **reference replay**: the expected value, returned by the root, is admitted by the runtime and accepted by the oracle;
3. **negative replay**: every `wrong_answers` entry is REJECTED by the oracle;
4. **alias replay**: every `alias_answers` entry is replayed and must be ACCEPTED (aliases are the oracle's alternates);
5. **smoke**: the `smoke.code` call graph runs with the scripted helper answers (catches wrong file placement, type
   errors, missing helpers); the smoke and reference scripts must also call each root helper with the declared number
   of arguments;
6. **custody**: no relation to any protected world (ProtectedWorlds registry) or the LFM3 held-set group closure;
7. **lint** (`scripts/rl_tasks/authored_checks.py`, static): `folder_reducer` or a `directory-reducer` root without
   `folders`; a readout that calls helpers or owns a folder; a skill bound to the wrong function; a
   `multi_operation_codebase` world with one root; `iterateOn` without a stated stop, or an `iterate` tag without
   `iterateOn`; `call_depth_3` without a chain or with a middle that never calls its sub-helper; helper calls spelled out
   in your instructions with the wrong argument count. Warnings prefixed `heuristic:` (pass-through or thin middle
   helpers, shared helpers across operations) are guesses for the auditor and fail nothing; take them seriously.

Qualification never runs the program as a model would, so none of this proves your `expected`: that is what the
literal run in section 10 is for.

Pass criterion: `"result": "PASS"` (every task passes 1-7). If some fail, fix or delete those tasks and rerun into
a fresh `qualified-2/` directory. Never weaken an oracle or delete a wrong answer just to pass: a wrong answer that is
accepted means the oracle cannot tell right from wrong, so the task is broken.

Then the repetition gate against the existing pool:

```sh
U=rlnov-<worker>-<batch>; rm -f $B/novelty.exit
python3 scripts/memory_ledger.py run --unit $U --budget-gb 12 --priority low --wait 3600 -- \
  sh -c "/home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/authored_novelty.py $B/qualified/verified/cases.jsonl \
  --pool /home/werg/data/broad-rl-tasks-20261031/full-core-effects-20261102-20261102T011031Z-3015925/cases.jsonl \
  --pool /mnt/external/natlang-rl-authoring/admitted/cases.jsonl -o $B/novelty.json > $B/novelty.log 2>&1; echo \$? > $B/novelty.exit"
until [ -f $B/novelty.exit ] || { ! systemctl --user is-active -q $U && ! systemctl --user is-active -q $U-waiter; }; do sleep 5; done
sleep 2; echo "job exit: $(cat $B/novelty.exit 2>/dev/null || echo MISSING)  unit result: $(systemctl --user show $U -p Result --value)"
```

A task is its program together with its inputs. Several invocations of one program on materially different worlds
and inputs are wanted (a multi-operation codebase is called many times). Pass: no exact duplicates (same program AND
same inputs after normalizing digits and quoted literals: swapping a number or quoted value is not a new task), at most
10% near duplicates (within a family, 5-word shingle Jaccard >= 0.8 for the programs AND for the inputs), and every
family either has 20+ distinct programs in total or gains 5+ new programs in this batch. Text is not enough: a program is also counted as a
*functional program* (structure: call graph, helper argument/return types, kind/readout, which helpers each function
calls, the code steps spelled out in backticks and the reference scripts, literals abstracted; plus the prose of the
helpers, at word-trigram Jaccard 0.7, so renaming the root or reformatting a sentence is not a new program but a changed
rule or a new operation is). Tasks whose program is functionally an existing one are listed under `functional_repeats`,
and in each family at least half of the batch's new programs must be functionally new (unless the family already has
20+ functional programs). New programs come from new operations, helpers, data flow or code steps, with inputs on which
the new rule changes the answer. `novelty.json` reports distinct programs and functional programs per family separately
from tasks (job exit 1 means FAIL, not a crash; read `novelty.json`; the file is written atomically, so a missing file
means the job did not finish: read `novelty.log`, never an older file). The functional count is a calibrated proxy
(10 of 12 audited round-1 batches within 30% of the auditors' distinct-program counts) and the audit still counts
behaviour: a rule sentence toggled inside one helper can look like one program to the gate and eight to the auditor.
Report both results to the supervisor; the supervisor moves passing batches into `admitted/` and records the rotation.

The `--no-resumable` flag is deliberately absent: the ledger backfills only resumable jobs behind a large waiting head,
and a qualification is simply rerun if preempted. Expect 30-50 minutes (queue, then the ~25 minute protection-closure
step, which is a fixed cost per run): pre-check with `authored_checks.py` and the converter
(`scripts/rl_tasks/sources/authored.py TASKS.jsonl`; it needs `alias_answers`, possibly `[]`) before queueing, and qualify once
with every fix. Round-2 pitfalls the verifier enforces: no `while` loops; one terminal `return` per reference script
(wrong answers replace only the last return); scripted reference children cannot replay `Neuralese<...>` helper
results (no write port), so tasks with Neuralese/typed-Neuralese helpers use the default direct-return reference (smoke
still scripts every helper) and their `expected` is verified only by the blind solve below.

**Blind solve and admission (supervisor).** After qualification and novelty PASS, the supervisor runs
`scripts/rl_tasks/blind_solve.py packet TASKS.jsonl -o PACKET` (strips expected, wrong answers, aliases, reference,
smoke, groups), has a different model solve every task from the packet alone (literal run, helpers see only their
arguments), and compares with `blind_solve.py compare TASKS.jsonl ANSWERS.json`. Every disagreement is settled by running
the program literally; every flagged ambiguity is sent back to the author. Admission is
`scripts/rl_tasks/admit_authored.py BATCH --qualified qualified-N --novelty novelty.json --reviewer ... [--drop ID]
[--supersede CASE_ID]` (appends under the pool lock, atomic, logged in `admitted/admissions.jsonl`), then
`authoring_rotation.py record ... --credit-from <batch>/qualified-N/verified/cases.jsonl`.

## 5. Exclusions

Never use as task inputs, seeds or inspiration text:
- anything in the ProtectedWorlds registry (`training/protected-worlds.json`) or registered with
  `admission: evaluation-only-never-training` in `training/neuralese_corpora.json`;
- `runs/natlang-dev-task-packet-v2-20261028` and every dev/confirmation/held packet (including
  `natlang-rl-broad-external-dev-20261031`, `natlang-rl-broad-external-confirmation-20261031`, the `held/` folder of
  any RL task source, LFM3 joint-SFT held sources);
- upstream evaluation splits (validation/dev/test) of any dataset.
The custody step catches related worlds; do not rely on it to catch copied text. When a dataset item seeds a task,
put its id in `groups` (e.g. `"hotpotqa:5a8b57f25542995d1e6f1371"`) so custody can relate it.

## 6. Three complete examples (qualified)

All three are in `/mnt/external/natlang-rl-authoring/examples/tasks.jsonl` (generator
`examples/make_examples.py`); their qualification report is `examples/qualified/report.json`.

### Example 1: iterate over Neuralese state + relay + exact state result (`stock_day_reconciliation`, level 2)

Constructs: `iterate_neuralese`, `neuralese_relay` (the carried note is passed into `absorb` and `net_movements`),
`effect_state`. The day's warehouse messages correct each other, so a carried private note is genuinely
useful; arithmetic stays exact in code.

`close_stock_day.nl` (root)
```
---
args:
  opening: "Record<string, number>"
  messages: "string[]"
returns: "Record<string, number>"
---
Close the warehouse day: return the stock count for every item in opening after all of today's messages.

Messages arrive in order and later ones can correct, cancel or add to earlier ones, so read them as one running story:
1. Keep one private running note. Start with `let notes = await begin_notes(opening);`.
2. Go through messages in order with iterateOn. Each step calls `notes = await absorb(notes, message);` for the next message, carrying the note forward. Stop when every message has been absorbed.
3. Turn the final note into movements: `const moves = await net_movements(notes, Object.keys(opening));`. Each movement is the net change of one item for the whole day.
4. Apply the movements exactly in code: start from a copy of opening and add each movement's change to its item. Return that record with every item of opening, including unchanged ones.
```
`close_stock_day/absorb.nl`
```
---
args:
  notes: "Neuralese<string>"
  message: "string"
returns: "Neuralese<string>"
---
notes is the private running account of today's stock movements so far. Read message and return the updated account:
- a delivery, return to us or found stock adds units; a pickup, shipment out or loss removes units;
- goods refused on arrival never entered stock;
- a correction replaces the earlier figure it refers to rather than adding a second movement.
Keep every earlier movement that message does not change.
```
`close_stock_day/begin_notes.nl` returns `Neuralese<string>`; `close_stock_day/net_movements.nl` takes
`notes: Neuralese<string>, items: string[]` and returns `{ item: string; change: number }[]`.

Inputs: `opening = {"oat milk": 40, "red lentils": 25, "rye flour": 18, "olive oil": 12}` and six messages
(12 cases delivered with 2 crushed and refused; 5 lentils picked up, later corrected to 3; 3 unlogged olive-oil
bottles found; 7 rye flour picked up, 2 returned). `expected = {"oat milk": 50, "red lentils": 22, "rye flour": 13,
"olive oil": 15}`, `oracle: "normalized"`. Wrong answers encode the real mistakes: counting the crushed cases (52),
applying the correction as an extra pickup (17), ignoring the return (11), ignoring found stock (12), dropping an item.
Alias: the same record with keys in another order.

### Example 2: decision readout + depth 3 + typed record (`warranty_claim_settlement`, level 3)

`handle_claim.nl` returns `{ route: "repair" | "refund" | "reject"; covered_cents: number }`. Steps: `decide(claim_kind,
claim, policy)` (a `readout: decision` helper with options `"manufacturing_defect" | "missing_or_wrong_item" |
"accident_or_misuse"`); reject on misuse; otherwise `assess_parts(claim, purchase, policy)` returns the covered part
names, and it calls its own sub-helper `assess_parts/coverage_window(part, policy)` (months of coverage) to compare
with the months between purchase and claim; the root sums `price_cents` in code. Inputs: a customer message about a
grinding rear hub and a cracked saddle, purchase 2026-03-14, claim 2026-10-02, policy text (hubs 12 months, saddles
6 months, tyres not covered). Expected `{"route": "repair", "covered_cents": 8900}` (hub only: the saddle's 6 months
have passed). Wrong answers: hub+saddle 13400, refund route, reject, frame+hub 51400. Alias: keys reordered.

### Example 3: typed Neuralese fan-in from three distinct producers (`event_venue_selection`, level 2)

`choose_venue.nl` calls, for every venue, three different assessors returning typed Neuralese:
`capacity_fit -> Neuralese<{ meets_hard_requirements: boolean; concerns: string }>`,
`budget_fit -> Neuralese<{ total_eur: number; within_budget: boolean }>`,
`travel_fit -> Neuralese<{ door_to_door_hours: number; needs_car: boolean }>`, then decides from all assessments.
Inputs: a brief (22 people, 2 nights, step-free, 9,000 EUR, train only, priority = shortest journey), four venues
(one has 20 beds, one needs a car, two qualify) and a travel sheet. Expected `"Haus am Kiefernsee"` (2 h vs 2 h 45 min),
`oracle: "normalized"`. Wrong answers are the other three venues (each fails for a different reason or loses on the
priority). Aliases: case and trailing-space variants.

## 7. Continuous authoring: two equal modes, rotation and fresh impulses

Authoring never stops after a quota. Every batch is in one of two **equal** modes; alternate between them.

**Mode A, dataset-inspired.** A local dataset (section 9) supplies realistic call arguments and targets. Take an
item's structure, question and numbers, wrap them in a program that genuinely needs the batch's construct, and compute
`expected` by running that program literally (section 10). The item's gold is a hint, never the answer: where the
program's routing, budget or rule removes the evidence the gold relied on, the literal answer differs and wins.
Cite the item id in `groups` only for items you actually used (question, numbers or gold); otherwise record the seed as
`invented:<domain>`. The dataset is a seed, not a constraint: change the framing, add state, corrections or operations.

**Mode B, original invention.** No dataset. Invent the program, its domain, its world data and its exact answer
yourself (compute the expected value by running your program literally, step by step, as section 10 describes; then write
wrong answers from the mistakes you nearly made). Techniques, use a different one each batch:
1. *Unusual real process*: pick a real-world workflow few datasets cover (beekeeping hive inspections, ferry
   car-deck loading, choir seating by voice part, archive accession numbering, darkroom chemical rotation, sports
   league tie-breakers, hospital linen par levels) and model its records and rules.
2. *Invent a stateful tool or service*: a booking ledger, a parts bin, a queue with priorities, a ration card, a
   tournament bracket; the task is operations on its state, and `expected` is the resulting state.
3. *Combine two unrelated domains*: a recipe scaler that must respect a shipping-weight table; a museum loan schedule
   under insurance rules; a garden planting plan driven by a bus timetable for the volunteers.
4. *Failure and recovery*: design what goes wrong (a duplicate payment, a corrected reading, a cancelled shift, a
   half-applied migration, a lost message) and make the task detect and repair it exactly.
5. *An algorithm as a genuine natlang workflow*: take an exact algorithm (topological order, interval merging,
   bin packing by rule, stable matching with stated preferences, shortest route on a given table, dedup by key) and
   put the semantic parts in NL (reading messy requests, judging which items match, deciding priorities) while the
   exact core stays in code. The semantic judgment must matter to the answer.
6. *Adversarial inputs*: near-duplicate records, misleading but irrelevant documents, an instruction buried in data
   that must not be followed, two rules that conflict with a stated precedence.
7. *Change the operator, keep the world*: a world built for one operation gets a second and third operation
   (report, undo, forecast, audit) to become a multi-operation codebase family.

Each batch takes one **cell** = (seed, construct, problem type), where seed is a dataset name (mode A) or
`invented:<domain>` (mode B). The rotation credits a batch to every construct it genuinely covers (`record --credit-from`),
not only to its cell's construct:

```sh
python3 scripts/rl_tasks/authoring_rotation.py next --ledger /mnt/external/natlang-rl-authoring/rotation.jsonl \
  --datasets /mnt/external/natlang-rl-authoring/rotation/datasets.txt \
  --constructs /mnt/external/natlang-rl-authoring/rotation/constructs.txt \
  --problems /mnt/external/natlang-rl-authoring/rotation/problems.txt --worker <worker> \
  --claim --batch /mnt/external/natlang-rl-authoring/batches/<worker>/<batch>
```

`--claim` reserves the returned cell for your batch, so other workers (and your own next batch) get a different cell.
`record` finalizes the claim. A claim not recorded within 6 hours (`--claim-hours`) expires. Always claim before
writing a batch.

`next` returns the least-covered construct, the mode that has admitted fewer tasks (A or B), and for mode A the
least-used dataset x problem cell; for mode B it lists every invented domain already used so you choose a domain
NOT in that list (and not a renaming of one). Concurrent workers are offset so they start on different cells. After
the supervisor admits a batch, record it (for mode B pass `--dataset invented:<your-domain>`; add
`--credit-from <admitted cases.jsonl of the batch>` or repeated `--credit construct=N` so every genuine construct is credited):

```sh
python3 scripts/rl_tasks/authoring_rotation.py record --ledger /mnt/external/natlang-rl-authoring/rotation.jsonl \
  --dataset <dataset or invented:domain> --construct C --problem P --worker W --batch DIR --admitted N \
  [--credit construct=N ...] [--credit-from CASES.jsonl]
```

**Steer away from the running ledger with corrections** (a list of notes folded into a running tally, later notes
correcting earlier ones): 5 of the 6 round-1 batches admitted that one problem type, and it is now over-represented.
Pick a computation of another kind: allocation or packing under constraints, routing over a table, scheduling with
precedence, matching with stated preferences, diffing two states, planning from a rule table, reconciling two sources,
bisection, graph or dependency order. Rotation priorities for the next round are `typed_neuralese`,
`neuralese_fan_in`, `folder_reducer` over real `folders`, and `skills` (0 to 4 admitted each).

Inside a cell, design a *program* first (a small application: its users, records/state, operations and failure
modes), then write 10-40 tasks against it across all four levels: different invocations, different worlds, different
corrections and edge cases. A new batch in an existing family must add new program variants (a new operation, helper
role or kind of input mess), not new numbers. Grow a family with new operations (undo, audit, carry-over, merge, repair,
forecast, counterfactual) or new helper roles; not with more note mixes, readouts of the same ledger, per-task rule
sentences switched on and off, or paragraphs that change no call, type or code step. For every rule variant include
inputs on which omitting the rule changes the answer, and say which input that is (a comment in the task's `shape` or
your batch notes). Padding added to pass the gate is removed in the audit and counted against the batch. The novelty gate (section 4) enforces this against everything admitted.

## 8. Gaps to fill (counts against the 2026-11-03 bar)

After round 1 (77 tasks admitted): `call_depth_3` 53, `iterate` 42, `effect_state` 32, `multi_operation_codebase` 14,
`decision_readout` 11, `iterate_neuralese` 8, `neuralese_relay` 8, `skills` 4, `typed_neuralese` 0, `neuralese_fan_in` 0,
`folder_reducer` 0. Author the zero and near-zero constructs first, with real depth, folders and skills.

After round 2 (520 tasks admitted in all, 27 families; pool `/mnt/external/natlang-rl-authoring/admitted/cases.jsonl`):
`effect_state` 264 (51%), `call_depth_3` 228, `iterate` 184, `decision_readout` 130, `typed_neuralese` 119,
`multi_operation_codebase` 98, `skills` 94, `neuralese_relay` 87, `neuralese_fan_in` 73, `folder_reducer` 61,
`iterate_neuralese` 49. Next round: grow the existing families to 20+ distinct programs each (every family has 6-15
today; the selection pool's share of families with 20+ programs fell from 67% to 50% because the authored families are new
and small), so write EXTENSION batches that reuse an admitted family name and add 7+ new programs; and take the constructs
still under 100 (`iterate_neuralese`, `folder_reducer`, `neuralese_fan_in`, `neuralese_relay`, `skills`,
`multi_operation_codebase`). Authored rows are 4.4% of the balanced selection pool (the generator-balance cap leaves them
their full supply), so the pool's effect/state share stays at 11%: reaching the 30% bar needs an authored (or generated)
effect-state supply of thousands of rows or an assembler quota, not more of the same batches.
Authoring pitfalls found in round 2 (all enforced by the checks or the blind solve): exact-oracle record keys and strings
must come from one stated convention (letter case, key names, parcel/part names: otherwise a solver and the author
disagree); folder worlds count in task identity; smoke/reference child fragments must be single-line and inside the first
two lines of the call text; mutants (`wrong_code`) that leave a world unchanged are accepted by the oracle, so every mutant
must change an effect on its world; assignments through the `store` input are forbidden (copy it to a `const`).

Bar: each construct >= 100 distinct qualified tasks; >= 70% of families with >= 20 distinct programs; >= 30% of the
selection pool with effect/state oracles. Existing RL source (61,399 tasks, measured with
`scripts/rl_tasks/measure_repetition.py`, `/mnt/external/natlang-rl-composed-20261103/repetition/before.json`):

| construct | existing tasks | notes | authoring target |
|---|---:|---|---:|
| `neuralese_relay` | 0 | conversion of multi-doc QA in progress (composed families) | 300+ authored in non-QA domains |
| `typed_neuralese` | 0 | same | 300+ |
| `neuralese_fan_in` | 0 (one-helper fan-in over documents only) | same | 300+ |
| `iterate_neuralese` | 0 | | 300+ |
| `iterate` (any) | 80 in 5 families | | 300+ |
| `call_depth_3` (distinct subproblems) | 5,500 are codebase questions with deep folders, not task call graphs | | 300+ |
| `folder_reducer` | 1,728, nearly all one repair template | | 200+ outside repair |
| `decision_readout` | 0 task programs (decision functions appear only inside codebase files) | conversion from labelled choice sets in progress | 200+ acting on probabilities (floors, abstain, route) |
| `skills` | 12 (one catalog) | | 200+ |
| `multi_operation_codebase` | ~664 single-template mutations | codebase-op conversion in progress | 400+ (worlds with 4-8 operations each) |
| `effect_state` | 1,550 (2.5%) | | as many as possible; every family should have state-result operations |

Genuinely different computational problems per construct (pick ones not yet in the ledger):

- **relay** (producer note handed to a different consumer): triage -> specialist (symptom notes to a dosage checker
  with a formulary table); reviewer notes -> rewrite (legal clause critique -> compliant clause, checked by required
  terms list); interview notes -> scoring rubric; bug report -> reproduction plan -> fix location; lab protocol notes
  -> reagent order quantities.
- **typed Neuralese**: per-document `{claim, supports, quote_id}`; per-account `{risk: "low"|"high", reason}`;
  per-route `{minutes, transfers}`; per-proof-step `{valid: boolean, rule}`; per-email `{asks_for, deadline}`.
- **fan-in**: several sensors/witnesses/reviewers each summarized, then a decision (sensor fault localization,
  conflicting witness timelines, multi-reviewer paper decision with explicit acceptance rule, multi-source price
  reconciliation, triangulating a meeting time from three calendars).
- **iterate / iterate over Neuralese**: running ledgers with corrections (stock, expenses, attendance, medication
  administration record), incremental schedule repair until no conflict, reading a long contract section by section
  with carried obligations, game state across turns (chess-like move lists, board games with explicit rules), bisection
  over versions with a provided test table, negotiation rounds.
- **depth 3**: plan -> per-step resolve -> per-fact lookup (travel itineraries with fare rules; recipe scaling with
  unit conversion tables; course prerequisite planning; tax form: income items -> category rules -> threshold lookup).
- **folder reducer**: a folder of receipts/tickets/log files/patient notes reduced to a summary state (expense report
  totals per category; incident timeline; open-issue list); conflicting files where later files supersede earlier ones.
- **decision readout**: gates acting on probabilities (refuse at p(unsafe) >= 0.7, escalate under 0.5 confidence),
  routing among teams, label-then-act chains, multi-decision aggregation (majority over three readouts).
- **skills**: a skills directory of 3-6 procedures (refund procedure, data-cleaning recipes, unit-conversion
  procedures, incident runbooks); the task needs choosing and applying the right one, checked by its exact outcome.
- **multi-operation codebases**: library loans (checkout/return/renew/fines report), clinic scheduling
  (book/cancel/reschedule/day sheet), small accounting ledger (post/reverse/close month/trial balance), course
  registration (enroll/drop/waitlist promote/roster), fleet maintenance (log/service due/parts order). Each world
  holds the initial state; each task is one operation's invocation; `expected` is the resulting state or report.

## 9. Dataset catalog (seeds for programs, arguments and targets)

See `plans/neuralese/RL_TASK_AUTHORING_DATASETS.md` (registry-derived catalog of local agentic and
knowledge-reasoning datasets, with schemas, protection status and 2-3 program ideas each). Use registries
(`training/neuralese_corpora.json`, `training/corpus-manifests/`, `/home/werg/data/external-rl-sources/SURVEY.md` and
each source's `PROVENANCE.json`); never crawl disks to find data. Prefer datasets marked underused.

## 10. Authoring discipline (from the round-1 audit)

Round 1 (163 tasks from six workers, `RL_AUTHORING_AUDIT_ROUND1_20261103.md`): 146 passed the audit, 99 of those only
after fixes, 17 were rejected, and 77 tasks from 6 batches were admitted. The failures were semantic, not arithmetic.
Do these things every time.

**Run the program literally.** Write out, for each helper call, exactly what the helper receives (its arguments only,
never the root's inputs or a preface it was not passed) and what its text makes it return. Apply the root's steps to
those returns. That result is `expected`. Typical misses: a router that sends the decisive snippet to the wrong bucket,
a budget that cuts off the fact, a gate applied to a file that never contained the fact. If the literal result differs
from what you intended, either change the program so it does what you intended or accept the literal result.

**One clearly right reading.** Name the entity in the note (box, barcode, hive id, ticket number). A reader who sees
"reel 2 confirmed" with no named item, or "artists sign their work" offered as proof that a named artist engraved it
(affirming the consequent), can argue either way; the task is rejected. If a recurring or worsening event must be
counted, state the counting rule in the helper that counts. Put every rule the answer needs in the text of the helper
that decides it.

**Drop a tag rather than simulate it.** `folder_reducer` means the task's `folders` and `expected_folders`, nothing else.
`multi_operation_codebase` means several root operations over shared helper files in one world. `skills` means a skill
that is the only place a needed fact lives, in the folder of the function that reads it. `decision_readout` means a
leaf. `neuralese_relay` and `iterate_neuralese` mean the relayed or carried note actually feeds the decision on the
branch the task takes (a relay that never runs when the answer is "ask" is not covered by that task). `iterate` means
several items and a stated stop. When unsure, remove the tag; `constructs: []` is fine.

**Real depth.** For `call_depth_3` the middle helper owns a different subproblem and combines its sub-helper's result
with its own work (for example, plan -> resolve one step -> look up one fact). A middle that forwards its arguments
and returns the sub-helper's value, or a sub-helper that is one comparison, is not depth. Exact comparisons and
arithmetic stay in code.

**Diversity through new computation.** A family grows by new operations, helper roles or data flow: undo, audit,
carry-over, merge, repair, forecast, counterfactual. It does not grow by toggled rule sentences, inert "Guard:" or
"Tie check:" paragraphs, or per-task prefaces the deciding helper never sees. For each rule variant, include inputs on
which omitting the rule changes the answer and know which task that is: in round 1 a refrigerated-first rule changed 2
of 8 answers, a crane reserve 1 of 7 and best-fit none of the level-3 tasks, so 16 tasks were 8 programs by behaviour.
Padding is removed in the audit and counted against the batch.

**Contract hygiene.** The question asks for exactly what the root returns (a sum when it returns a sum, not "the
average"; a count when it returns a count, not "which"). Use exactly representable decimals (halves, quarters, tenths
only where the program rounds explicitly): JavaScript gives `5 - 4.2 = 0.7999999999999998`, and the exact oracle expects
`0.8`. Otherwise state the rounding in the root's instruction. Wrong answers are task-specific mistakes (a missed
correction, a wrong branch, an off-by-one), never a copy-pasted "colour: blue". State the counting rule for recurring
events. Give every `iterateOn` an explicit stop.

**Honest levels.** Label by what the task asks of a solver. Write genuine level-3 tasks in every family: long inputs,
a dozen or more items, corrections that interact, or competing constraints with a stated precedence. A two-call relay is
level 0 and a one-call filter is level 0 or 1.

**Honest provenance.** Cite `groups` only for dataset items you used. If you borrowed only a format (a plan layout, a
record shape), say `invented:<domain>`.

## 11. Self-audit before qualifying

The audit solves each task independently from the program text and inputs alone, with `expected`, wrong answers,
aliases, reference and smoke stripped out, and then compares. Do that to your own batch first, and answer every line
honestly for every task. Any "no" means fix or delete the task, not argue it.

1. Did I simulate each helper with only the arguments it receives, then write `expected` from that run (not from the
   dataset gold and not from what I meant)?
2. Is every fact the answer needs in the inputs or in the text of the helper that decides it, with no root-only rule?
3. Does every judgment have one clearly right reading, with the entity named in the note?
4. Does the question ask for exactly what the root returns, in the unit and rounding it states? Are the decimals
   exact?
5. For each construct tag: can I point at the file and line that makes it true (section 3)? If not, remove the tag.
   `folder_reducer` has `folders`; the readout is a leaf; the skill sits in its reader's folder and holds a needed fact;
   the middle helper has its own subproblem; `iterateOn` runs over several items and states its stop; every spelled-out
   helper call has the declared argument count.
6. For every rule variant in the batch, which input changes its answer when the rule is removed? Is there such an
   input for each?
7. Is each task a new computation, or the same program with a sentence switched, a name or number changed, or an inert
   paragraph added? Count the distinct programs by behaviour; that is what the audit counts.
8. Are the levels honest, with real level-3 tasks, and is each wrong answer a real, task-specific mistake that the
   oracle rejects?
9. Is `groups` limited to items I actually used, with `invented:<domain>` otherwise?
10. Does the batch avoid the running-ledger-with-corrections problem type, and does it cover one of the priority
    constructs (`typed_neuralese`, `neuralese_fan_in`, real `folder_reducer`, `skills`)?

Then run section 4.

## 12. Round 3 changes (2026-11-03)

- **Smoke replay is required for Neuralese tasks.** A task claiming `typed_neuralese`, `neuralese_relay`, `neuralese_fan_in`
  or `iterate_neuralese` sets `"smoke": {"code": ..., "children": ..., "replay": true}`. With `replay` the smoke code is the
  WHOLE root script (every item, every helper call, the exact code steps) and its returned value is judged by the oracle
  against `expected`, so the arithmetic/filtering/ordering behind `expected` is checked by a real run with the helpers'
  scripted answers (the structure smoke runs the teacher-text Neuralese transport). Script each helper per call with
  `{"$match": [{"match": "<text in its call, single line, in the first two lines>", "value": ...}, ...]}`. A Neuralese
  value is opaque to code: code cannot read `.field` of a `Neuralese<{...}>` result (the run stops), it can only pass the value
  into another helper. So put the exact steps after the Neuralese stage in a code step over the PLAIN return of a consumer
  helper (e.g. a `net_movements(notes, items)` returning a record array), and give each scripted helper the answer its text
  would literally produce. Where the final decision is taken in language by the root from typed Neuralese records, the replay
  cannot check it (the blind solve still does): add a final plain helper or exact code step wherever the answer has one.
  Qualification fails Neuralese tasks without `replay`, and warns when the replay returns a bare literal.
- **Protection closure is seconds, not 25 minutes**: it now uses a persistent exact-signature hash cache
  (`/mnt/external/natlang-rl-authoring/protection-cache/hash-cache.jsonl`); the union digest is recomputed from the actual
  protection inputs every run and recorded in `report.json` (`steps.protection_closure`).
- **Selection pool quota**: the assembler keeps every effect/state row under the family cap and shrinks the other rows until
  effect/state oracles are 30% of the pool (no duplicated rows). Each admitted effect/state task therefore grows the pool.
- `scripts/rl_tasks/authored_family_report.py FAMILY` lists the programs a family already has (extension batches).

## 13. Round 3 results and operating notes (2026-11-03)

Round 3 ran 25 Haiku-authored extension batches (w39-w68), each through the full chain: qualify, novelty, independent blind
solve by a Sonnet subagent from the packet alone, admission. Lessons that the blind solve found (all were semantic, not
arithmetic), now rules for every batch:
- A readout (`readout: decision`) sees only its arguments: pass it every fact its label depends on (pharmacy: the drug schedule;
  a limit's direction; a drug name). A label that two careful readers could split is dropped, not argued.
- State every output convention in the program text: ordering keys, id forms ("Room 101" even if the note says "room 101"),
  number formats in written files, band edges ("0.5 or more but less than 0.8"), what a trailing "?" means.
- Folder tasks: the blind solver must also reproduce `expected_folders`; give the solver a folder-state answer format.
- Tasks whose blind answer differs or is flagged ambiguous are held back at admission (`--drop`), together with their program
  when the defect is in the program text; admitted counts below exclude them.
- Run ledger commands detached (`setsid nohup`, or the ledger `--detach` flag): a foreground ledger client that exits stops
  its unit ("on client request"), which killed several novelty runs.
- `multi_operation_codebase` needs shared helper files; the lint warns when roots share none (round-3 library/clinic worlds
  keep per-root helpers: the tag there rests on shared world state, treat as weaker).
- Pool assembly: `assemble_selection_pool.py --spec pool-spec.json` now takes `min_effect_share` (0.3), `program_caps`
  (per external source, e.g. bgkit 40 rows per program) and `exclude_ids`.

## 14. Round 4 changes (2026-11-03)

- **Effect/state means a world.** `measure_repetition.program_features` counts `effect_state_oracle` only when the oracle verifies a
  world after the run: effect keys (`effects`, `expected_files`, `files_oracle`, `function_checks`, `operation`, `service_scopes`,
  `codebase_check`), `folder_files`, or a service whose state the oracle reads. An author-declared `oracle_depth: effect-state`
  with none of that evidence is `returned_state_oracle` (a record computed from the arguments): kept, reported separately, never
  counted toward the owner's 30% bar. Round 3's assembler share (30.0%) had counted 511 such authored rows; the strict share of the
  same 6,666-row pool was 22.4% (1,490 rows). An independent count that excludes the 100 authored folder-effect rows reproduces
  20.9% (1,390 rows). Author `effect_state` tasks as `folder_reducer` worlds with `expected_folders`.
- **Pool assembly**: train rows related to any registered protected packet or held source are dropped (receipt:
  `protected_world_related`, `protected_worlds_digest`); `min_pool_rows` (pool-spec) bounds the quota's shrinking and the receipt
  reports `quota.met` and the honest share instead of dropping breadth; returned-state rows are kept whole but not counted.
- **Held authored rows** are registered with `scripts/rl_tasks/register_authored_held.py` (immutable stamped packets
  `natlang-rl-authored-{dev,confirmation}-<stamp>`, evaluation-only, pinned manifests). Run it after every admission wave, commit
  the manifests and `training/neuralese_corpora.json`, then rebuild the pool (the protection digest changes). Authors reused
  `invented:<domain>` group strings across worlds that the world split put on different sides, so 13 train rows were related to
  held rows and are never selectable.
- **Rotation credit**: `record --credit-from` credits only the batch's admitted rows (admission log); `reconcile` rewrites a
  ledger to the admission log (25 entries had over-credited tasks the audit dropped).
- **Blind solve**: every packet row carries an `answer_format`; folder tasks must list EVERY file of the final world folder with exact
  text. `blind_solve.py prompt` prints the uniform solver prompt; the solver returns `{"answers": ..., "ambiguous": ...}` and
  `compare` fails on any ambiguity. Round 4 found one wrong `expected` per roughly 40 tasks that qualification could not see
  (a root that never calls its helper because the crew list omits the names; a total of 1300 for a literal 1360).
- **multi_operation_codebase lint is an error** without an identical helper file (same name, same text) in two or more roots' folders.
- **Producer notes**: `admit_authored` attaches `reference.children` for Neuralese producer helpers from replay-verified smoke
  scripts (`authored_child_notes.py`; per-call `$match` values, never constants).
- **The family field is the bare family name**; the converter adds `auth_`.

### Round 4 results and operating notes

- Second-wave batches (w100 to w112) were all folder worlds with `expected_folders`; admitted counts after the Sonnet blind solve:
  w100 11/16, w102 16/16, w104 14/16, w105 18/18, w106 15/16, w107 18/19, w108 16/16, w109 17/17, w110 17/20, w111 13/16,
  w112 8/12 (w103 admitted 15 of 16, w85 never qualified, the 16th w103 row was not admitted because the batch-level novelty
  gate matches the rows already in the pool). Dominant drop reasons: an unspecified output format the solver had to guess
  (comma-list separator, singular/plural wording, line separator), a count that includes or excludes a case the text leaves open,
  and judgment boundaries in readouts. The blind solve is cheap (about one minute per batch) and catches what qualification cannot.
- The effect/state bar was met honestly by authored folder-world supply, not by shrinking the pool: strict share 22.4% at 6,666 rows
  before, 30.0% after the wave, with `min_pool_rows` 6,666.
- Novelty key gap: the exact-duplicate key digit-normalizes folder contents and file names, so folder-only variants of one program
  (`01.md`/`02.md` entries, a long and a short world) can be flagged as duplicates. Authors work around it with letter file names;
  round 5 should hash the folder file names and the normalized content explicitly.
- Authors should state every written line format (separators, singular and plural wording, trailing newline) in the root, and the
  count a returned number includes; these were the commonest blind-solve rejections.
