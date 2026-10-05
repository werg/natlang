# Rich recurrence corpus — 2026-10-05

## Findings

The v13-r2 native teacher file contains 111,301 decisions, 45,940 eval targets,
6,905 child decisions, 2,122 successful child `return_result` targets, and 847
text/structured return targets at least 16 characters long. Only 1,388 eval
targets contain inline-lambda syntax (~3%); 560 match named-function call syntax.
These are lexical candidates, not successful invocation counts. Count targets,
not repeated history. There are also 121 plain child replies (69 long): the
converter and trajectory writer currently support tool-shaped final writes only.

The restrictive semantic prompt originated in `df1f679a` (Oct 1). It survived
subsequent additions of lambda examples. The user requested its removal and an
increase from three to **five** ad hoc layers. Current source promotes inline
judgments, extraction, transformations and itemwise calls. At layer five, prompts
and help remove creation guidance; layer six is refused. Every file-backed `.nl`
function starts a new root budget. Frozen campaigns retain their pinned prompt;
new campaigns need a rebuilt, verified runtime before these changes take effect.

Conversion v4 recognizes named functions from pinned program files and scopes
result names by trajectory identity plus value, preventing cross-run collisions.
A no-output audit on the same teacher file raises counted writes 153→197 and
reads 342→582. These are occurrences including history, not unique producers.
Remaining counts: producer-missing 2,020; crisp-value 248; value-not-printed 1,336.
Newly recognized calls expose additional omissions. Do not invent missing returns
or treat formatting differences as evidence of execution. Equal values within a
run can still have ambiguous producers. Native export now preserves invocation
IDs from the raw trajectory for subsequent precise recovery.

## Implementation and live work

- `inline-curriculum/recurrence.mjs`: exact-oracle triage, delivery evidence and
  cancellation policy; private leaf services, nested named functions, paired
  hidden worlds with differing answers and identical parent openings. Both named
  graphs and graphs with anonymous semantic judgment lambdas are registered.
- Depth 1–4, width 1–3, varying source padding. Padding size is **not** verified
  model context length: runtime pagination can shorten what the model sees.
- Source siblings share split groups; held-out seeds differ from training seeds.
- `verify-recurrence-pool.mjs` checks holds, runtime replay, native linking against
  a specified collector runtime. Proof is not model-output admission.
- `audit_natlang_delegation.py`: streaming actual-target frequency audit.
- `audit_neuralese_recurrence.py`: producer closure, ambiguity, cycles, graph/source
  split checks. Missing historic compaction producers can legitimately use crisp
  source fallback; distinguish those from missing child writers before admission.
- 300 initial named cases and 48 held-out cases verified. Both 32-case named and
  inline pilots verified. 600 expanded training cases across both families verified with clearer
  once-per-helper wording and a separate seed. Published source-only snapshot:
  `recurrence-task-pool-20261005-v1` (plus48 held-out cases), not model trajectories.
- Space Bunny 32-case named pilot is running in
  `runs/recurrence-expansion-20261005/space-bunny/`, one request, free Stealth only,
  no distillation flag, normal retry delay/backoff. Frozen v38 runtime. At review,
  20 completed / 3 accepted: root duplicated a child call in an inspected failure.
  Keep raw failures; do not loosen exact array/type/label grading or scale blindly.
- Local writer-control GPU run reached step 200/512 with zero terminal errors;
  no inference job competes with it. DGX job ownership remains with the DGX agent.

## Next required work

1. Finish pilot rejection clustering (wrong labels, duplicated children, incorrect
   result shape, scope mistakes). Verify native admission and observed call graph,
   not collector completion alone. Register sealed accepted outputs separately.
2. Rebuild/freeze a current runtime with the new prompt and five-layer budget;
   replay proof, then a fresh inline pilot. Coordinate external/DGX scheduling.
3. Recover complete valid historical child finals from raw traces using preserved
   invocation identity; exclude ambiguity and protected split crossings. Support
   plain final writes in converter **and trainer/template rendering together**.
   Match structured displayed values using the runtime's exact printer, not fuzzy
   whitespace/text matching. Escaped JSON strings also need exact treatment.
4. Add shared-child reuse, argument/instruction handoffs and deeper inline chains;
   extend to grounded summaries, translation/editing and skill improvement.
   Current three small constructed domains are a starting calibration suite.
5. Implement correct/shuffled/zero/removed-return and counterfactual evaluation.
   Controls currently declared in case metadata are **not an implemented evaluator**.
   Measure content dependence and task pass rates, not merely lower target CE.
6. Build a closed, graph-group-split recurrence training snapshot after admission,
   publish SHA manifests, synchronize, and train/evaluate on that snapshot. Do not
   count case pools, reference replays, replicas and generated trajectories as
   separate independent data.

Evidence: `training/audits/delegation-20261005.json` and
`training/audits/recurrence-conversion-v4-20261005.json`.

## Directory reducers: separate semantic coverage from the reducer mix

The recorded full v13 training mix has31.5256% reducer decisions, but this is not
an inline-use quota. Many source directory tasks are exact edits/computation;
CommitPack alone has2,649 eval targets in the delegation audit. Semantic folder
families are sparse (`folder_triage`:39 child returns / one inline eval target;
`folder_mixed`:one eval target). Short semantic labels also stay crisp under the
converter's16-character rule, so Neuralese writer counts cannot measure lambda
utilization. Track actual executed children per root case separately.

Existing folder families mark delegation required only at100files; 20–45file
cases make it optional and140–200file cases are expensive teacher campaigns.
New `folder_triage_inline` and `folder_index_inline` families use8–24files and
explicit semantic lambda judgments followed by exact moves/counts. Their source
identities, source partitions, noise filtering and file oracle thresholds are
unchanged. Both families are in the shared registry;8pilot cases (4base +4hinted
siblings) replayed successfully, published as source-only snapshot
`semantic-folder-inline-pilot-20261005-v1` and synchronized to DGX.

Reducer-specific prompt now promotes inline per-file judgments/extraction/
transformations; depth-limited variant omits that advice. Focused prompt tests
cover both surfaces. Next: rebuild collection runtime and run a small pilot,
then expand semantic file classification, evidence extraction, rewriting, joined
records and nested subfolder reducers. Preserve the25%overall reducer target and
report semantic-reducer case share, lambda decisions, successful child invocation
counts, result types, and actual context sizes separately. Do not equate a short
label's lack of a soft writer with lack of delegation.

## Static inline/iteration trajectory expansion

User requests collection tasks with runtime-selected per-item operations, scoped
`natlang.d` libraries, focused extraction from large inputs, file/folder work and
multi-step semantic `iterateOn` with runtime-composed instructions. DGX agent owns
new runtime-composed workbench generators: semantic collections/decisions/folder
reducers and semantic iteration optimization, helper distractors and an operation
grammar, hidden exact labels. Pop retains recurrence and conversion; do not fork
competing workbench implementations.

Existing reference replay already supplies a no-model path. Pop exercised seed6110
across inline_multi_capture, inline_union_target, inline_structured_extract,
named_versus_inline, iterate_schedule_repair and composed_process:30admitted
trajectories,140native turns,80child-return targets. Native export approves58
individual decisions (16inline eval targets,10iterateOn targets);82decisions stay
held, including scripted direct answers and redundant actions. Synthetic action
notes are explicitly not trained as reasoning. Immutable registered snapshot:
`static-inline-iterate-reference-pilot-20261005-v1`, synchronized to DGX. Still
apply source/split closure when joining the full dataset; replicas are not more
examples and held child answers must not silently become direct-answer targets.

Promising source/task patterns:
- runtime policy predicates over labeled collections; exceptions/negation/joined
  evidence; exact moves/counts after NL classification;
- query/schema-derived extraction from documents and nested folder reports;
- nearest library helper vs a newly composed predicate/extractor, including helpers
  that solve only part of the request and must be combined with an inline lambda;
- per-file rewriting/localization with preserved facts/front matter and constraints;
- progressive evidence search with question updates after conflicting observations;
- iterative draft/plan/program/skill repair with actual checker feedback and a
  measured objective; next instructions depend on observed violations;
- semantic deduplication and record resolution using controlled hidden identities.

For static generation: derive gold from source annotations or a constructed world,
script permitted actions, execute actual tools/children/iterations, and admit the
observed result. Do not fabricate tool results or chain-of-thought. For open-ended
rewrites use cheap proposal generation plus executable constraints and independent
semantic grading; provenance and uncertainty remain explicit. Vary controller,
helper library, objective, feedback, depth and graph topology—not just nonce IDs.
Use held-out policies/source groups and counterfactual inputs to detect memorized
scripts. Iterative examples must carry intermediate state and feedback, and should
include useful alternative routes and honest no-improvement stops rather than
claiming an imposed monotonic sequence demonstrates general optimization skill.

## 2026-10-05 static expansion and independently composed folder operations

Registered `static-lambda-expansion-20261005-v1`: **1,272 train cases**, 10,225
native decisions, 9,137 approved and 1,088 held. This includes 540 constructed-world
workbench cases, 24 labeled-source folder cases, 660 inline/iteration cases and
48 cross-source folder cases. There are 722 approved inline eval decisions, 433
iterateOn eval decisions and 6,635 unique child invocations. Invocations are not
necessarily nested depth: wide loops also account for these counts. The older
stock-note workbench replay is superseded, not another training corpus.

Separate `static-lambda-heldout-20261005-v1` has 108 evaluation-only cases and
1,036 decisions. Both snapshots preserve raw references, verification receipts
and filtered native exports. These representations are not additive examples.
Native-to-neuralese conversion, graph closure and global source/split closure
remain required; local correctness approval does not waive them.

`cross_source_folders` independently chooses an authored operation recipe, a
semantic criterion, SMS spam / SST-2 / Banking77 source records, weights and
subfolder layout. Its first48 examples cover all12 operation–dataset combinations:
selection index, weighted total, document routing, and iterative selection. The
agent reads runtime task.json and actual documents; source labels are host-only.
Near-miss library helper `mentions_money` does not decide the actual criterion.
No answer labels are embedded into documents or filenames. Gold correctness is
still limited by source annotations: those classification return targets stay
held; exact reads/calls/file operations are admitted. Source IDs remain available
for de-duplication and split closure across other recipes using the same records.

Course decisions:
- User authorized short authored action plans as supervision. Explicit two-field
  provenance (`authored-action-plans/1` plus `authored-action-plan`) enables that
  supervision; ordinary scripted action notes stay masked. Plans contain intended
  actions, not hidden answers or invented observations. 26 scoped materializer
  tests passed during implementation.
- Exact oracle returns in constructed-world workbench/iteration examples are
  explicitly admitted with `--supervise-reference-answers`. Dataset-derived
  semantic answers remain held; do not blanket-enable the option.
- The user also authorized synthesizing deterministic tool results directly into
  IR. That faster path is not implemented here: these batches execute real runtime
  tools, with scripted oracle child responses and zero model requests. A future
  direct-IR route must identify synthetic results honestly, pin inputs, verify
  state transitions and preserve invocation/producer identities.
- Iterated selection uses an exact decreasing remaining-document measure. Its
  scripted progress review was initially missing; that replay bug was fixed,
  rather than changing runtime review policy. Failed partial staging directories
  are unsealed evidence and excluded from published snapshots.
- Generator summaries preserve actual HEAD at generation. Some runs preceded
  committing local implementation; the post-generation implementation receipt
  names commit0a2627ca without pretending the original checkout was clean. New
  builds additionally capture generator source hashes before execution.

DGX independently added dataset-workbench, claims, contract and translation desks;
merged those changes from main. Continue task expansion without duplicating its
owned generation jobs. A recipe × data product creates variation, not independent
source evidence; evaluate on held-out source records and held-out policies too.

Conversion follow-up: registered `static-lambda-neuralese-20261005-v1`, derived
from the train native snapshot, with all10,225records. Audit found28writer records,
5reader records,1unambiguous edge and8ambiguous producer links; whole output stays
held. Crisp boolean returns intentionally do not become soft values, and inline
NL-literal softening is currently deferred by converter curriculum. Thus the large
call count proves executed inline usage, not rich soft return recurrence. Expand
structured extraction/rewritten-text returns and fix invocation-aware producer
matching; do not discard raw native evidence or silently admit ambiguous graphs.
Native train/evaluation snapshots were SHA-verified on DGX after transfer.

### 2026-10-05 reviewed conversion/5 and richer recurrent packet

The earlier conversion/4 audit above remains historical evidence. New immutable
`static-lambda-neuralese-20261005-v2` recovers319writers,97readers and319edges
from the same10225native records. Invocation identity, observed parentage and
canonical/runtime-printer matching fix producer collapse and missed links.
No ambiguous/missing/cyclic/cross-split graph edges remain, but source/target
quality still controls admission; this output remains a candidate.

New independently reviewed `clef-rich-reviewed-cohort-20261005-v1` contains
1120train/747held records,670writers,557readers,677edges,depth5,branch3.
78train and52held programs derive from12 versus8 authored semantic fixtures;
permutations/width/depth variants do not count as new facts. Sources, exact
choice receipts, runtime replay/admission checks and fixture isolation are
registered separately and synchronized to DGX. Families are skill-catalog
selection/read and nested quote extraction; they do not establish broad
workflow competence. Executable inline lambdas are present in native replay;
conversion still leaves their instruction literals crisp. Soft inline
instruction tuning is a remaining curriculum step.

Actual checkpoint256 tests failed all8 autonomous sampled tasks and all4
conditional returns in every ablation. This exposed the frozen interpreter
policy gap. Pop now runs mixed-policy-v1 (2048steps) warm-started from joint
best512, with trainable rank16 LoRA and sequential exact-text SFT replay,
periodic crisp/written/shuffled evaluation and adaptive recurrence staging.
Initial updates fit and have zero errors; downstream task improvement remains
unproven until new held-out execution results arrive. Long-context testing
from the prior20.3k stage must be retained alongside new deeper4.7k graphs.
