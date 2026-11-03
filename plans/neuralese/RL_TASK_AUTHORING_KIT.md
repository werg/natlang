# RL task authoring kit (2026-11-03)

For model workers who author new RL tasks for the LFM3 student. Owner decision (2026-11-03): fill the missing target
behaviours mainly with on-policy RL, so the student needs many genuinely different natlang programs with exact
oracles. You write programs, inputs, exact answers and controls. You never write model trajectories.

Read this whole file before writing anything. Rules that override everything else:

- **Every fact the answer needs is visible in the inputs or the program.** No hidden gold. If a careful human with
  the inputs could not get exactly the expected value, the task is wrong.
- **The answer contract is visible.** The root's `returns` type and its instruction say exactly what form the answer
  takes (record fields, units, whole numbers, exact names).
- **Real diversity, not renaming.** A new task must differ in what is computed, its call graph, its state, or its
  failure modes. Swapping names, numbers, products or cities into the same program is the *same* task; the novelty
  gate rejects it.
- **Never use protected or held data** (see Exclusions). Never copy text from evaluation packets.
- **Policy is natural language; exact work is code.** NL helpers interpret, judge and summarize; arithmetic, sums,
  sorting, counting and state updates happen in the root's code (`eval`) or are spelled out as exact steps.
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
| `family` | snake_case name of the *program design* (the reusable application/program), e.g. `stock_day_reconciliation`. All tasks that run the same program design share one family. Aim for 20+ genuinely different tasks per family over time. |
| `shape` | short name of the call-graph pattern, e.g. `carried-notes-then-exact-apply` |
| `domain` | free word: `operations`, `planning`, `finance`, `science`, `customer_service`, `code`, ... |
| `world` | identity of the scenario world. Tasks that share facts (same store, same document set, same codebase, same customer) MUST share `world`; the split (train/dev/confirmation) is derived from it, so related tasks never cross splits. |
| `level` | 0-3 real difficulty: 0 = one helper call and a direct answer; 1 = two or three steps; 2 = several dependent steps or carried state; 3 = long inputs, many items, corrections or competing constraints. Use all four levels in a family. For constructs that need at least two helpers (relay, fan-in, depth 3), level 0 is the simplest two-call version: short clean inputs, no corrections, each helper used once. |
| `constructs` | non-empty subset of: `call_depth_3`, `neuralese_relay`, `typed_neuralese`, `neuralese_fan_in`, `iterate`, `iterate_neuralese`, `folder_reducer`, `decision_readout`, `skills`, `multi_operation_codebase`, `effect_state` (definitions in section 3) |
| `root` | the root file name, e.g. `"close_stock_day.nl"` (top level, ends in `.nl`) |
| `files` | object path -> file text. The root plus its helpers. Helpers callable by the root live in the folder named like the root without `.nl` (`close_stock_day/absorb.nl`). A helper called by a helper lives in that helper's folder (`handle_claim/assess_parts/coverage_window.nl`). Skills: `root_folder/skills/<skill-name>/SKILL.md`. |
| `inputs` | object: the root's arguments by name, with concrete values |
| `expected` | the exact correct result, matching the root's `returns` type |
| `oracle` | `"exact"` (deep equality; records compare by field, key order irrelevant) or `"normalized"` (also ignores case, surrounding spaces and number formatting for strings/numbers) |
| `oracle_depth` | `"effect-state"` when `expected` is the full resulting state of the world after the task's operations (a stock ledger, a schedule, a set of bookings, a repaired config); otherwise `"answer"`. Must agree with the `effect_state` construct. |
| `wrong_answers` | 2-6 *meaningful* wrong results of the right type: what a model gets by a real mistake (missed correction, wrong branch, off-by-one, dropped item, applied rule in the wrong order). Each must differ from `expected` and must be rejected by the oracle. |
| `alias_answers` | 0-3 alternative forms of the SAME correct result that should be accepted (reordered record keys, case/spacing variants for normalized). Must be type-valid. |
| `smoke` | `{"code": "<root eval script>", "children": {"<helper name>": <scripted answer>}}`. The code calls the helpers the way the program intends (one pass is enough); each helper answers with its scripted value. It proves the call graph wires and type-checks. Typed Neuralese helpers answer with a record; `Neuralese<string>` helpers with text; `readout: decision` helpers with one of their options. |
| `groups` | optional list of extra world/source identities (e.g. the dataset item id your task was inspired by) |
| `reference` | optional; default is a direct return of `expected` |

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
| `call_depth_3` | root -> helper -> sub-helper, each solving a *different* subproblem (not the same task delegated again) |
| `neuralese_relay` | a helper returns `Neuralese<...>` and that value is passed as an argument into another helper |
| `typed_neuralese` | a helper returns `Neuralese<{ ...record... }>` |
| `neuralese_fan_in` | two or more Neuralese-producing helpers (distinct roles, or one per item) whose outputs the parent combines |
| `iterate` | the root uses `iterateOn` over items/steps with an explicit stopping condition |
| `iterate_neuralese` | `iterateOn` whose carried state is (or contains) a Neuralese value |
| `folder_reducer` | a helper with frontmatter `kind: directory-reducer` that reduces over a folder of items |
| `decision_readout` | a `readout: decision` helper whose distribution the parent acts on via `decide(...)` |
| `skills` | `skills/<name>/SKILL.md` files the program loads and applies to choose or perform work |
| `multi_operation_codebase` | the task's root is one of several operations of the same reusable program (same `family`, same helper files, different root functions across tasks of one world) |
| `effect_state` | `expected` is the complete resulting state, checked field by field |

## 4. Qualify your batch (mandatory)

Run on the DGX through the memory ledger (one command; it converts, replays, smoke-runs, checks custody and writes a
report):

```sh
cd /home/werg/natlang
B=/mnt/external/natlang-rl-authoring/batches/<worker>/<batch>
python3 scripts/memory_ledger.py run --unit rlauth-<worker>-<batch> --budget-gb 4 --priority low --wait 3600 -- \
  env PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang \
  /home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/qualify_authored.py $B/tasks.jsonl $B/qualified
until ! systemctl --user is-active -q rlauth-<worker>-<batch> && ! systemctl --user is-active -q rlauth-<worker>-<batch>-waiter; do sleep 5; done
cat $B/qualified/report.json | head -60
```

What it checks per task:
1. **convert**: the JSON matches section 2 (errors name the line and field);
2. **reference replay**: the expected value, returned by the root, is admitted by the runtime and accepted by the oracle;
3. **negative replay**: every `wrong_answers` entry is REJECTED by the oracle;
4. **alias replay**: every `alias_answers` entry is replayed; acceptance is recorded (a rejected alias is a warning:
   fix the alias or change `oracle` to `normalized` when the variant is genuinely the same answer);
5. **smoke**: the `smoke.code` call graph runs with the scripted helper answers (catches wrong file placement, type
   errors, missing helpers);
6. **custody**: no relation to any protected world (ProtectedWorlds registry) or the LFM3 held-set group closure.

Pass criterion: `"result": "PASS"` (every task passes 1-3, 5, 6). If some fail, fix or delete those tasks and rerun into
a fresh `qualified-2/` directory. Never weaken an oracle or delete a wrong answer just to pass: a wrong answer that is
accepted means the oracle cannot tell right from wrong, so the task is broken.

Then the repetition gate against the existing pool:

```sh
python3 scripts/memory_ledger.py run --unit rlnov-<worker>-<batch> --budget-gb 6 --priority low --wait 3600 -- \
  /home/werg/natlang/.venv-neuralese/bin/python scripts/rl_tasks/authored_novelty.py $B/qualified/verified/cases.jsonl \
  --pool /home/werg/data/broad-rl-tasks-20261031/full-core-effects-20261102-20261102T011031Z-3015925/cases.jsonl \
  --pool /mnt/external/natlang-rl-authoring/admitted/cases.jsonl -o $B/novelty.json
```

A task is its program together with its inputs. Several invocations of one program on materially different worlds
and inputs are wanted (a multi-operation codebase is called many times). Pass: no exact duplicates (same program AND
same inputs after normalizing digits and quoted literals: swapping a number or quoted value is not a new task), at most
10% near duplicates (within a family, 5-word shingle Jaccard >= 0.8 for the programs AND for the inputs), and every
family either has 20+ distinct programs in total or gains 5+ new programs in this batch. `novelty.json` reports
distinct programs per family separately from tasks.
Report both results to the supervisor; the supervisor moves passing batches into `admitted/` and records the rotation.

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
item's structure and gold, wrap it in a program that genuinely needs the batch's construct, and keep the item id in
`groups`. The dataset is a seed, not a constraint: change the framing, add state, corrections or operations.

**Mode B, original invention.** No dataset. Invent the program, its domain, its world data and its exact answer
yourself (compute the expected value by hand, step by step, and double-check it; then write wrong answers from the
mistakes you nearly made). Techniques, use a different one each batch:
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
`invented:<domain>` (mode B):

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
the supervisor admits a batch, record it (for mode B pass `--dataset invented:<your-domain>`):

```sh
python3 scripts/rl_tasks/authoring_rotation.py record --ledger /mnt/external/natlang-rl-authoring/rotation.jsonl \
  --dataset <dataset or invented:domain> --construct C --problem P --worker W --batch DIR --admitted N
```

Inside a cell, design a *program* first (a small application: its users, records/state, operations and failure
modes), then write 10-40 tasks against it across all four levels: different invocations, different worlds, different
corrections and edge cases. A new batch in an existing family must add new program variants (a new operation, helper
role or kind of input mess), not new numbers. The novelty gate (section 4) enforces this against everything admitted.

## 8. Gaps to fill (counts against the 2026-11-03 bar)

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
