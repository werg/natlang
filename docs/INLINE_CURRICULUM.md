# Inline-natlang curriculum: cases, verification, and admission

This is the working implementation of the [ancillary data plan](INLINE_NATLANG_TRAINING_DATA_PLAN.md). It builds
interpreter-track cases (`natlang.program/2` projects) whose correct trajectories place inline `nl` lambdas
well, or decline to, and whose later choices depend on observations the opening does not show. Code lives in
`ts-host/src/teacher/curriculum.ts` (schema, verification, admission, coverage) and
`ts-host/scripts/inline-curriculum/` (families, source adapters, build, acquisition, admission CLI).

## Case format

A case is an ordinary program IR record plus a `curriculum` block (`natlang.inline_curriculum/1`). The collector
gives the model only `semantics`, so the block is oracle metadata the teacher never sees:

| Field | Meaning |
|---|---|
| `family`, `shape`, `variant` | Generator family, program shape, and input variant |
| `pair_group` | Counterfactual group: the variants share one visible opening and differ in a hidden observation |
| `split_group` | Every variant of one underlying problem (or one source story) shares a split group |
| `slice`, `domain` | The plan's behavior slice and reasoning domain, for balance |
| `mode` | `single_call` (one well-formed eval can be right) or `followup` (a later choice depends on an observation) |
| `inline` | `required`, `optional`, or `avoid` (a gratuitous child) |
| `edits` | `required` (a real defect in a callable helper) or `forbidden` (a correct helper) |
| `named`, `iterate` | `required`: the trajectory must call a named callable `.nl` function, or run `iterateOn` |
| `decisive` | Markers of the observations the answer rests on, with where they first appear |
| `plausible_actions`, `minimum_sequence` | What could be done before the observation; the causally sufficient sequence |
| `evidence`, `assumptions`, `world_semantics` | World assertions, what must be retrieved, bridging knowledge, open/closed/defeasible semantics |
| `reference` | A replayable solution: root tool calls, and answers for child calls keyed by fragments of the child's opening (a value, or a tool call such as `blocked`) |

Results are exact typed values (labels, ids, receipts, certificates), so a semantic decision is still checked
exactly. Worlds are callable-folder TypeScript modules: data stays hidden until the model queries it, module
state persists across evals, and each function's doc comment is shown in the opening's function listing.

## Build-time verification

`build.mjs` writes a shard only when every case verifies (`verifyCases`):

- the record and its curriculum block validate;
- no decisive marker appears in the rendered opening (system prompt, instructions, and pre-filled scope);
- the reference solution replays through the collector's own execution path (`executeProgram`) and is admitted;
- the variants of each pair group render byte-identical openings and do not all expect the same result
  (result, blocked/failed outcome, and required edits all count as differences).

These checks caught real defects during construction: a page count and file sizes in the opening that told
variants apart, a random second draw that repeated the first answer, a callable export named `apply` (a
reserved function property), and loop bounds the eval policy rejects.

## Admission

`admit.mjs` scores collected rows (`admitRow`) and writes a ledger and coverage. A row is admitted when the
collector accepted its contract **and**:

- every decisive marker was visible to the model (in a tool result, or in a child it delegated to) outside the
  root's opening, and, for follow-up cases, before the root's first result decision (a `return_result`,
  `blocked`, `failed`, or a staged eval `return`);
- `inline: avoid` saw no inline child. For `inline: required`, a correct answer judged directly is admitted with the
  note `judged_directly`; an eval that tests text against a keyword regular expression instead is rejected
  (`regex_judgment`);
- `edits: required` saw an `edit_function`, and `edits: forbidden` saw none;
- `named: required` saw a named child call, and `iterate: required` an eval that runs `iterateOn`.

Rejection reasons: `wrong_return`, `fabricated_result` (a value where the honest outcome is a blocker),
`incomplete_trajectory`, `missing_observation:<marker>`, `premature_choice:<marker>`, `inline_missing`,
`gratuitous_inline`, `regex_judgment`, `defect_not_repaired`, `unwarranted_edit`, `named_helper_unused`, `iterate_missing`.
An eval that failed (nothing kept) does not count as a result decision, even if it contained a `return`.

### Hinted twins and preference pairs

Every `iterate: required` case is also built as a twin whose instructions end with an explicit requirement to use `iterateOn`
(`build.mjs`, on by default). Admission strips the hint from an admitted twin's trajectory and program, so the row
trains `iterateOn` unprompted. `pairs.mjs` pairs an admitted twin with the unhinted run of the same case when that
run was rejected for `iterate_missing`: at their first root decision with an identical request, the twin's
decision is chosen and the unhinted one rejected (the same-request rule of `build-preference-pairs.mjs`). The number of evals is never a criterion.
The collector itself now requires a blocked case to end with the model's own `blocked` or `failed` call;
running out of turns also quiesces a call and was previously accepted.

## Family catalog

| Family | Slice | Domain | Mode | What varies across the pair group |
|---|---|---|---|---|
| `logic_entailment_exception` | follow-up | logic | follow-up | A late exception applies to the entity, to a decoy, or the membership fact is missing; a third of shapes use real kinds whose local rule contradicts common knowledge |
| `logic_proof_verifier` | follow-up | logic | follow-up | A guarded rule's negated condition holds, is defeated by a late fact the verifier names, or a needed link is absent; a proof returns the verifier's certificate |
| `logic_abduction_test` | follow-up | logic | follow-up | A discriminating test confirms the suggested cause, reverses it, or is inconclusive |
| `folio_entailment` | follow-up | logic | follow-up | FOLIO premises in a paged store mixed with an unrelated story's; one case per example |
| `folio_batch` | inline | logic | single call | Every conclusion of one FOLIO story, each judged in its own inline child |
| `prontoqa_proof` | follow-up | logic | single call | A PrOntoQA proof as a chain of fact ids checked by a verifier; the counterpart lacks a rule the proof needs |
| `prontoqa_search` | iterate | logic | single call | The same proofs found by forward search with `iterateOn` |
| `kqapro_question` | nested | relational | single call | A KQA Pro question over a paged knowledge-base module holding what its gold program touches, plus decoys |
| `proofwriter_question` | follow-up | logic | single call | Open-world true/false/unknown over a paged theory; the counterpart removes a fact the only proof uses |
| `entailment_premises` | follow-up | logic | single call | Premise selection for a science hypothesis among distractors, checked by a verifier; the counterpart lacks a needed premise |
| `anli_batch` | inline | logic | single call | Five αNLI stories, each judged in its own inline child |
| `commaqa_question` | nested | relational | single call | A CommaQA question answered through a table specialist and a text specialist |
| `commaqa_numeric` | nested | relational | single call | CommaQA numeric: per-item specialist questions, with the arithmetic (min, max, differences, thresholds) left to the root |
| `textworld_quest` | follow-up | actor | single call | A generated TextWorld quest played through look/commands/act; the counterpart lacks the needed object (blocked) |
| `textworld_iterate` | iterate | actor | single call | The same quests played by an inline nl step run with iterateOn; without the object, the progress review stops the loop |
| `scienceworld_task` | follow-up | actor | single call | A ScienceWorld task (boiling, chemistry, biology, ...) played through the `world` service until the score reaches 100 |
| `alfworld_task` | follow-up | actor | single call | An ALFWorld household task (find, heat, clean, place, examine) played through the `world` service |
| `child_sufficiency` | nested | logic | follow-up | A named child finds the first records sufficient, needs the detailed records, or neither suffices (blocked) |
| `relational_multihop_qualifier` | nested | relational | single call | Paged two-hop graph with temporal qualifiers: base, divestment, a hire on a later page, a departure |
| `relational_policy_inline` | inline | relational | single call | Exact candidate retrieval, then per-candidate policy judgments through a typed `review_each` callback |
| `relational_dynamic_snapshot` | follow-up | relational | follow-up | Prose events since a cached snapshot transfer, remove, leave unchanged, or undo a change |
| `actor_route_replan` | follow-up | actor | follow-up | Taking the item closes the short route, closes both safe routes (blocked), or nothing happens; a hazard rule forbids a shortcut |
| `iterate_schedule_repair` | iterate | other | single call | `iterateOn` with zero steps, a few, a long run with progress reviews, or an oscillation the judge stops |
| `inline_cohort_policy` | inline | other | single call | Rows past the preview make the per-group policy verdict supported, contradicted, or uncertain, including a Simpson reversal |
| `inline_review_each` | inline | other | single call | A per-item semantic filter; which tickets report a live problem |
| `inline_avoid_crisp` | inline | other | single call | The same helper and inbox with an exact field criterion: an inline child is gratuitous |
| `paged_late_row` | follow-up | other | follow-up | A late health check past the preview shows a regression, a recovery, or an internal-only error |
| `contract_diagnosis` | failure | other | follow-up / single | A helper violates its documented contract (fix it) or meets it (leave it) |
| `folder_criteria_reducer` | failure | other | follow-up | A directory reducer's criteria file selects different tickets to move |
| `scoped_module_discovery` | nested | other | single call | Nested callable leaves (`pricing.discounts.seasonal.percent`) with a flat-name trap |
| `policy_after_measure` | follow-up | other | follow-up | A runbook policy read after measuring p95 sets a threshold or a maintenance exception that does or does not cover the spike |
| `counterexample_revision` | follow-up | other | follow-up | A late record shows that a shared email is a household inbox (revise the rule) or the same person (keep it) |
| `parallel_labels` | inline | other | single call | Concurrent `nl<Label>` children whose labels must stay attached to their items |
| `inline_late_binding` | inline | other | single call | One saved inline judgment captures a `let` budget reassigned between screening rounds |
| `inline_type_repair` | inline | other | follow-up | A seeded eval reads fields of an untyped inline lambda's result; the diagnostic proposes the annotation, which leads to a typed lambda (or a direct answer) |
| `idempotent_retry` | failure | other | follow-up | A send succeeds but its acknowledgement is lost; the same command is retried under its key |
| `live_inventory` | nested | actor | single call | A live class instance from a module: reserve every line, or roll back on a short line; the stock fingerprint proves the rollback |
| `inline_multi_capture` | inline | other | single call | Exact per-service metrics and release notes judged together under a policy: notes announce the failures, are unrelated, or announce another path |
| `inline_union_target` | inline | other | single call | A union-of-records triage per message, collected with `Promise.all` and keyed back by id |
| `stateful_dates` | inline | other | single call | A `Map` of `Date`s: overdue contacts, with a note deferring one to a date that has or has not arrived |
| `inline_structured_extract` | inline | other | single call | Each receipt's total and currency need a typed record (`nl<Amount>`), then exact sums per currency |
| `named_versus_inline` | nested / inline | other | single call | The folder's `urgency` function fits (a new inline function is gratuitous), or no helper fits (judge inline) |
| `loop_rewrite` | failure | other | follow-up | A seeded `while` loop is rejected: rewrite as a counted loop (page count known) or with `iterateOn` (until an empty page) |
| `recursion_rewrite` | failure | other | follow-up | A seeded recursive org-chart depth is rejected; rewrite with an explicit bounded walk |
| `reducer_apply` | failure | other | follow-up | Release notes decide which package changed code; `bump_version` is applied to that package's folder only |
| `child_opt_out` | nested | other | single call | One invoice states no total: its child reports blocked and the parent records null, never a computed figure |
| `event_retry` | failure | actor | follow-up | A committed event whose view times out is rendered again, not re-applied; a rejected event is reported |
| `iterate_frontier` | iterate | relational | single call | Breadth-first search link by link with `iterateOn`: near, far (past the progress review), or unreachable |
| `relational_late_argmax` | follow-up | relational | single call | The supplier with the most late Q3 shipments; late, other-quarter, or cancelled rows on the last page decide it |
| `actor_greenhouse` | iterate | actor | single call | A controller stepped with `iterateOn` until the temperature holds: sun arrives, a cold start runs past the review, or a broken heater makes it blocked |

## TypeScript authoring track

Authoring rows (`curriculum.track: "authoring"`) are directory-reducer calls that edit a small project so that a
module implements a function with `nl`, a named callable function, or `iterateOn`. `semantics.authoring`
(`ts-host/src/teacher/authoring.ts`) judges the result by running it: the edited folder is mounted as a callable
folder, the export is called on test inputs, and its natural-language children are answered by an oracle keyed
by fragments of the child's opening (argument values), since the author's instruction wording is not known in
advance. The spec can also require the source to create an `nl` function, use `iterateOn`, call a named
function, or create no `nl` function where an existing one fits. `select.mjs` takes authoring rows only with
`--track authoring`.

| Family | What is authored |
|---|---|
| `authoring_inline_review` | A per-ticket semantic filter with an inline `nl` inside the module |
| `authoring_iterate` | A link-distance search stepped with `iterateOn` (checked on near, far past the progress review, and unreachable) |
| `authoring_named_helper` | An aggregate that must call the folder's `sentiment.nl`, not a new judgment |
| `authoring_structured_extract` | Per-currency totals whose per-receipt judgment needs a typed result (`nl<Amount>`) |
| `authoring_captured_policy` | A release gate: exact error rates, then an inline judgment that captures `policy` by name |

Building the checker found that `import type { X } from "./types"` in a callable module resolved to nothing,
so the aliases became `any` and inline children lost their types; module compilation now provides the folder's
aliases as an importable `types` module.

## External sources

`acquire.mjs` fetches pinned files into the untracked `vendor/datasets/` cache, checks recorded checksums, and
writes `data/teacher/inline-curriculum/sources.manifest.json` (URL, revision, checksum, split, size). Adapters
read only the cache and keep source labels and formal annotations in the oracle block.

- **FOLIO** is pinned at GitHub revision `5d7bb84` (v0.0, MIT). The corrected v2 release on Hugging Face is
  gated behind accepting its terms with an account; switch the source to it once that is done. v0.0 spells the
  third label both `Uncertain` and `Unknown`, and has the label errors v2 fixed, so its cases need review
  before training. Validation stories become `test` cases.
- **PrOntoQA-OOD** (Apache-2.0) is pinned at `0a6412b` (`generated_ood_data.zip`, regenerated 2024-10-17). The
  proof-only files give 4,100 distinct theories, all of which parse and derive their goal; in-context examples
  are `train`, the benchmark's test examples `test`.
- **KQA Pro** is pinned at the Hugging Face mirror `drt/kqa_pro@0b26da6` (the maintainers' download link no longer
  serves the archive). The authors license it **CC BY-SA 4.0**; share-alike may extend to derived training
  rows, so review this before training on them. Each question's gold program is translated into TypeScript
  over a paged `kb` API and run against the full knowledge base; only questions whose run reproduces the
  dataset answer are used, and the translated program is the reference. Validation questions are `test`.
- **ProofWriter** (CC BY 4.0, V2020.12.3, open-world depth-5): deep questions with a single proof, paired with the
  same theory missing a fact that proof uses (so the statement becomes unknown). Dev and test theories are `test`.
- **EntailmentBank** (CC BY 4.0, task 2 with distractors, via the Hugging Face mirror
  `ariesutiono/entailment-bank-v3`; the official copy is a Google Drive folder): select the premises of a
  hypothesis, checked by a verifier that accepts the gold premises plus at most two others; paired with a needed
  premise removed.
- **αNLI** (Apache-2.0, data-only archive): five stories per case, each judged by its own inline child (which
  hypothesis better explains the ending); one hypothesis pair per story.
- **CommaQA** (Apache-2.0, explicit v1): each question is answered through two named specialists, a table expert
  and a text expert, each holding half of a world's evidence; the decomposition's step answers are their
  reference answers. In the numeric variant the specialists do no arithmetic: `project` steps become one
  specialist question per item, and the root computes minima, maxima, differences, and threshold filters.
- Sources whose maintainers publish no checksums had them recorded on first acquisition and pinned in
  `acquire.mjs`. The manifest (`data/teacher/inline-curriculum/sources.manifest.json`) is committed.
- **TextWorld** (MIT, textworld 1.7.0): 300 games generated by `tw-make custom` with seeds 1-300 (objective
  states only the last action) and exported by `textworld_export.py` with the game's own action rules, types,
  facts, and win condition. A callable module runs them with an iterative rule engine, so preconditions and
  effects are exactly TextWorld's; every game's own winning commands are checked to win. 236 games have a
  counterpart without the object the win condition needs (provably impossible, so blocked).
- **ScienceWorld** (Apache-2.0, scienceworld 1.2.2, pinned below 1.3.0 whose action ordering changed): the
  simulator runs in its own process (`scienceworld_bridge.py serve`) and is the program's `world` service
  (`src/teacher/world-bridge.ts`); a case is accepted when the task's score reaches 100. Variations 0-2 of all 30
  tasks (90 cases; variation 2 is `test`) have gold action paths from the package, and all 90 replay to 100.
  Collection and verification need `SCIENCEWORLD_PYTHON` (default `vendor/scienceworld-venv`) and Java 11+.
- **ALFWorld** (MIT, alfworld 0.4.2, json_2.1.1 games): the same live bridge (`alfworld_bridge.py`), a won game
  scoring 100. 285 games (up to 100 per split, spread over task types; valid splits are `test`) whose handcoded
  expert wins; its command list is the reference.
- Acquisition extracts only the archive members an adapter reads (ProofWriter: OWA depth 5, 0.5 of 3.4 GB).

## Commands

```sh
npm --workspace @natlang/typescript-host run build:node
cd ts-host
node scripts/inline-curriculum/acquire.mjs --source folio
node scripts/inline-curriculum/build.mjs --seed 1 --shapes 2 --out ../data/teacher/inline-curriculum/smoke-s1.ir.jsonl
node scripts/teacher-collector.mjs ../data/teacher/inline-curriculum/smoke-s1.ir.jsonl ../runs/ic.jobs ../runs/ic.results.jsonl \
  --model-id Ternary-Bonsai-2-27B --root-seed 909 --server http://127.0.0.1:8081 --all --workers 1 --max-turns 30
node scripts/inline-curriculum/admit.mjs ../runs/ic.results.jsonl --ledger ../runs/ic.ledger.jsonl --admitted ../runs/ic.admitted.jsonl --show
```

`--shapes` scales every family by its weight; `--start` offsets the shape index so shards are disjoint.
`build.mjs --split test` marks a build's synthetic cases as held out (use a seed no training build uses; sources keep
their original splits). `select.mjs POOL.jsonl --out SHARD.jsonl [--split train|test] [--track authoring] [--no-balance]` picks the largest shard whose domain and slice
shares are all within three points of the plan's targets, keeping counterfactual groups whole and spreading
picks across families: from a 2,580-case pool (seed 102, `--shapes 20`, about 30 s) it selects 1,455 cases
from all 39 families, the largest family at 7%. With 53 families, the seed-102 pool (`--shapes 20`, 3,268 cases,
47 s) gives a 1,702-case train shard over 48 families; a seed-900 held-out build plus the sources' test splits
gives a 310-case test shard; the authoring track has 100 cases.
Synthetic case ids and groups carry the seed; source cases are grouped by source story across shards.
Admitted rows are ordinary collector rows and go through `materialize-native-teacher.mjs` unchanged. (Pilot 4's 37
admitted rows gave 169 training decisions.) Materialized rows keep the program IR, oracle block included, as
provenance; `export-native-sft.mjs` builds model input from `messages`, `tools`, and `target` only, so no oracle
field reaches training input.

## Model-surface changes made for this curriculum

- The system prompt explains inline `nl` in eval (when to create one, how its type is given, what it sees, and
  when not to) and `iterateOn` for open-ended repetition. Both were previously undocumented to the model.
- The opening's function listing shows each function's doc comment (`.nl` `description`, TypeScript JSDoc).
  Before, a helper could not say how its pages are numbered or what its result means.
- A callable module's type aliases are in eval's type environment (they were shown in the listing but
  unknown to eval), its classes are live class types, and its interfaces are records or live shapes. A class
  is listed as a TypeScript declaration of its public members with their doc comments.
- Natlang type text accepts single-quoted string literals, so `const risk: 'low' | 'high' = await nl`...`` works.
- An unannotated inline `nl` gets its result type from how the eval uses it (typed holes, `compiler/holes.ts`):
  conditions, `Promise.all(items.map(...))` results, arithmetic, typed variables, and `iterateOn` steps and stopping
  checks no longer need `nl<T>`. Before, 17 of 20 natural unannotated uses were rejected.
- In eval, `await nl`...`` without a call runs the judgment (a live run awaited the function itself, so every
  verdict was truthy); project source gets an `nl-not-called` diagnostic instead.
- A staged result is shown in full when it is small portable data. A live run saw `Staged { reserved: …, … }`,
  retyped the value into `return_result`, and invented the reservation ids it had never been shown.

## Pilot findings

Pilot 4 (Bonsai 27B, 80 cases, one counterfactual group per family, before the families added with it):
18 of the first 21 admitted, including every logic case and the relational multi-hop cases whose edge
direction pilot 3 misread. Rejections were a policy review that awaited `nl` functions without calling them
(fixed in the surface), a review whose "judgment" was keyword regexes (a correct rejection), and the invented
reservation ids above.

