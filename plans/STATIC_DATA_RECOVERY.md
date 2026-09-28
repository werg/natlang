# Static data recovery audit — 2026-09-28

Modern recovery is now implemented; see [MODERN_STATIC_ADAPTERS.md](MODERN_STATIC_ADAPTERS.md)
for actual admission, holds and audits. The inventory below is the earlier recovery
checkpoint. It does not override the newer conservative source policy.

## Finding and scope

Valuable source data survived on disk, but several adapters did not survive the
Python retirement. The shell SFT builder and staged production recipe also differed
in which existing static trajectories they included.

Compared both builders, runs/source-backed-20260928/recipe.json, preserved
source/IR snapshots, history before 8a6e44a, and fresh native replays. This covers
20 named snapshots, not every ignored file or historical teacher job. Counts
overlap across source tasks, programs and snapshots; **do not sum them as distinct
approved examples**.

Reproduce with `python3 scripts/audit_static_data.py --output runs/static-data-inventory.json`.
Use `--recipe FILE` to audit a saved recipe. The streaming report records content
hashes, a recipe hash, formats, splits and direct input membership. Missing direct
wiring does not prove a skill is absent from freshly generated cases.

## Repairs and actual recovery

| Gap | Evidence | Change |
| --- | --- | --- |
| Static references omitted from staged recipe | ref-v1: 792 trajectories / 2,999 decisions; ref-composed-v1: 1,400 / 7,229 | Default admission/materialization stages now feed an existing-curriculum track into teacher preparation and joint assembly. |
| Retired prompts included by default | Six code turn snapshots total 120 rows; IR /1 and obsolete read_value, mark_lines, report_blocker, report_error tools | Hold old turn snapshots; replay eligible saved captures on the frozen runtime. Explicit existing verified-turns inputs also require IR /2. |
| Helper discovery used boxed lambda | Current root is a filename; recipe looked for root.$lambda.codebase | Require current IR, matching source layout and actual helper files. Inlined/missing/recursive graphs remain held. |
| Replay changed source arguments | Async arrow inherited eval wrapper arguments | Use a real function invocation with the source parameter list; conversion version function-scope-v3. |
| Runtime failures missing from rejection ledger | Range replay produced 20 failed trajectories but zero reported rejections | Log unsuccessful runtime/equality outcomes as well as thrown conversion errors. |

Current admission accepts all **2,192 reference trajectories**. Materialization
produces 10,228 evidence decisions, of which **5,174 are approved**. Held: 4,970
direct scripted answers, 80 failed/unexecuted proposals, 4 checker-rejected attempts.
These counts precede preparation deduplication, split assignment and Sharp's final
8,192-token audit; they are not a completed ready corpus.

Fresh capture replay recovered descending (8 trajectories / 16 turns) and reverse
(1 / 2). Range remains held (20 attempts): parameter assignments and while conflict
with current eval rules. Dependency-bearing captures require retained workspaces
and separate helper projections; they are not silently flattened.

Evidence lives in runs/legacy-data-audit-20260928/: final-recipe.json,
final-inventory.json, reference.admission.jsonl, reference.turns.jsonl, and
verified/{descending,reverse,range}.jsonl with adjacent rejection ledgers.
No model requests or training were started for this audit.

## Valuable sources still requiring adapters or review

| Source on disk | Observed size | Recovery needed |
| --- | --- | --- |
| NanoJev curated train tasks | 1,272 decisions: 424 choice, 424 boolean, 424 score | Typed decision adapters with criteria and grouping. Wider 3,182-task file includes extra game families; curated tasks are a subset. |
| Typed Decisions pilot | 15 train questions | Same adapter; synthetic labels need review. This is not the full upstream dataset. |
| Sales classifier outputs | 17,900 accepted labels: 14,286 train / 3,614 dev; 100 HTTP errors | Decision/prefix/fold adapters. API acceptance is not correctness: 8,641 labels have recorded confidence below 0.8. Confidence alone is not an admission rule. |
| FinQA | 6,036 train programs / 9,343 arithmetic steps | Executable numeric adapter, explicit operand references, calculation/source-answer agreement and evidence before arithmetic. Preserve 845 dev programs. |
| SCONE | 11,198 train sequences / 55,990 annotated transitions | Typed state reducers, chain continuity, domain rules and intermediate-state gold. Preserve CC-BY-SA-4.0 and 642 dev sequences. |
| Schema-Guided Dialogue | 29,642 train service sequences / 175,780 transitions | Slot/intent reducers, schema metadata and prior dialogue context. Service-local changes cannot imply other state was cleared. Preserve CC-BY-SA-4.0 and 4,570 dev sequences. |
| CLEVR | 7,000 train scene programs | Symbolic scene/operator adapter with executable checks; preserve 1,500 validation records. |
| Old synthetic composed/leaf mix | One 10,000-program snapshot: 8,299 graphs / 1,701 sources, 23 families | Migrate types, instruction semantics and references; execute on current runtime. Generic IR migration drops old scripted references. Snapshots overlap. |
| Reviewed leaf reference bank | 620 entries | Small-function input/output pairs need typing, modern prompts and review/provenance joins. Retain earlier quarantine decisions. |
| Legacy teacher decision archive | 1,713 unique archived rows / 6,426 decisions; manifest records 50 recovered program revisions | Recovery evidence, not current-interface trajectories. Preserve original choices/failures; do not fabricate modern reasoning or promote benchmark traces. |

### Qualifications

- direct-core-v2.ir.jsonl (1,939 programs) and sales-all-v2.ir.jsonl (22,419)
  still contain IR /1 and unsupported decision kinds. Filename v2 does not mean
  natlang.program/2.
- All 1,900 sales top-up IDs already occur in the consolidated 18,000-row file.
  Appending both duplicates labels. Preserve company split groups across prefixes.
- jeff's 1,600 benchmark tasks are intentionally test-only. SQL has 100 saved
  decisions across train/dev/test and a prior source/teacher disagreement; it needs
  executable/source-label review, not blanket teacher acceptance.
- The 300 migrated native algorithm programs do not imply lost family coverage:
  current teacher seeds recreate those three shapes. Existing 512-program
  native-synthetic and coverage-selection snapshots still use IR /1; prefer fresh
  generation and compare family coverage before appending historical revisions.
- The new source-backed directory pilot is already connected to both builders:
  51 cases / 157 approved decisions with a completed Sharp token audit. It does
  not restore the older decision/state/scene adapters.
- The staged recipe now imports the two static reference sets automatically.
  Other saved Bonsai/Luna collections need completed-job snapshots passed with
  repeatable --teacher-results, or the shell builder's explicit input list.
  Selected-range exports omit other job directories. Fresh collection stages do
  not prove that every historical completed job is included.

## Recovery order

1. Decision adapters plus the reviewed leaf-bank join: short samples and composed
   map/count/report programs from grouped sources. Scripted gold answers remain
   synthetic/direct; default reasoning SFT holds them unless the student lane
   explicitly permits direct answers.
2. FinQA and SCONE using arithmetic and intermediate-state oracles for longer
   structured trajectories without model generation.
3. Dialogue-state and scene adapters, then compare legacy synthetic family coverage
   against current tracks. Preserve context, holdouts and provenance.
4. Eligible teacher decisions with explicit conversion scope; do not replace
   original reasoning or conceal changed tasks/environments.

Verification: Node build, browser bundle/types, focused materializer/source/replay
tests (38 passed), Python recipe/preparation/runner tests (26 passed), reference admission/materialization,
and three actual capture replay attempts. Older corpora above remain migration
candidates, not newly approved data. Bonsai and the single Luna repair worker
continue on their existing frozen runtime.
