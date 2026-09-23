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
| `decisive` | Markers of the observations the answer rests on, with where they first appear |
| `plausible_actions`, `minimum_sequence` | What could be done before the observation; the causally sufficient sequence |
| `evidence`, `assumptions`, `world_semantics` | World assertions, what must be retrieved, bridging knowledge, open/closed/defeasible semantics |
| `reference` | A replayable solution: root tool calls, and answers for child calls keyed by a fragment of the child's opening |

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
- `inline: required` saw an inline child, and `inline: avoid` saw none;
- `edits: required` saw an `edit_function`, and `edits: forbidden` saw none.

Rejection reasons: `wrong_return`, `fabricated_result` (a value where the honest outcome is a blocker),
`incomplete_trajectory`, `missing_observation:<marker>`, `premature_choice:<marker>`, `inline_missing`,
`gratuitous_inline`, `defect_not_repaired`, `unwarranted_edit`. The number of evals is never a criterion.
The collector itself now requires a blocked case to end with the model's own `blocked` or `failed` call;
running out of turns also quiesces a call and was previously accepted.

## Family catalog

| Family | Slice | Domain | Mode | What varies across the pair group |
|---|---|---|---|---|
| `logic_entailment_exception` | follow-up | logic | follow-up | A late exception applies to the entity, to a decoy, or the membership fact is missing; a third of shapes use real kinds whose local rule contradicts common knowledge |
| `logic_proof_verifier` | follow-up | logic | follow-up | A guarded rule's negated condition holds, is defeated by a late fact the verifier names, or a needed link is absent; a proof returns the verifier's certificate |
| `logic_abduction_test` | follow-up | logic | follow-up | A discriminating test confirms the suggested cause, reverses it, or is inconclusive |
| `folio_entailment` | follow-up | logic | follow-up | FOLIO premises in a paged store mixed with an unrelated story's; one case per example |
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
| `live_inventory` | nested | actor | follow-up | A live class instance from a module: reserve every line, or roll back on a short line; the stock fingerprint proves the rollback |

## External sources

`acquire.mjs` fetches pinned files into the untracked `vendor/datasets/` cache, checks recorded checksums, and
writes `data/teacher/inline-curriculum/sources.manifest.json` (URL, revision, checksum, split, size). Adapters
read only the cache and keep source labels and formal annotations in the oracle block.

- **FOLIO** is pinned at GitHub revision `5d7bb84` (v0.0, MIT). The corrected v2 release on Hugging Face is
  gated behind accepting its terms with an account; switch the source to it once that is done. v0.0 spells the
  third label both `Uncertain` and `Unknown`, and has the label errors v2 fixed, so its cases need review
  before training. Validation stories become `test` cases.

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
Synthetic case ids and groups carry the seed; source cases are grouped by source story across shards.
Admitted rows are ordinary collector rows and go through `materialize-native-teacher.mjs` unchanged.

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
