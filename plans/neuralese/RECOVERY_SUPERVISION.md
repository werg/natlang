# Recovery supervision: a changed action after a no-progress observation (design, 2026-10-13)

Source: plans/neuralese/REPETITION_DRAGNET_20261013.md, recommendation 4. Both students loop: the same action and the
same observation for 8 to 127 turns. The training targets almost never contain a repeated action (0.08–0.10% of
supervised turns repeat a call on an identical observation), so the model never sees "this did not move, so do
something else". This family teaches exactly that turn. It is data for union r4/r5 (both lines; records are
tokenizer-agnostic) and adds no runtime prompt text (teach in errors, not prompts).

## What a training example is

A trajectory prefix that ends in an observation showing no progress, followed by the supervised next assistant turn,
which is a **different** action that makes progress. The no-progress turn stays in context and is never a target.

No progress, decided crisply and exactly on the observation (no model judgement needed):
1. **Identical:** the observation's bytes equal an earlier observation of the same action in the trajectory (after the
   runtime's own normalisation: the `Stored local` line drops on a rerun).
2. **Empty:** a successful read that returned an empty collection, `null`, or an empty string (the multihop
   `graph.outgoing(parent, "owns", 1)` → `[]` loop).
3. **Failed:** a runtime failure form (`FAILED_TOOL_FEEDBACK` in train/text_supervision.py: rejected action, discarded
   eval, refused result). These turns are already context-only in training.

## Three sources (in order of cost)

1. **Mined natural recoveries (zero cost).** Scan the registered native-turn corpora (r3/r4 renders, both renderings)
   for supervised turns whose previous observation is no-progress (1–3 above) and whose action differs from the
   previous action. These exist (errors followed by fixes, empty reads followed by the other direction) but are a
   small share. Register them as a sampling group (`recovery`) so the direct trainer's CohortDocumentSampler gives them
   their own weight, as the owner asked for the agentic tool-loop families (handover §4, union r4). Output: a
   selection manifest of (document, turn) pairs over existing corpora, not copies.
2. **Exact repeat-then-change branches (CPU, exact).** For a supervised turn t+1 whose context ends with action a_t
   and observation o_t, insert k ∈ {1, 2} reruns of a_t before it and take each rerun's observation from the real
   runtime (`ts-host/src/native/runtime.ts` replay of the trajectory with the call repeated), so the observation bytes
   are exactly what serving returns (including the existing failure-repeat note where it applies). The target stays the
   teacher's own a_{t+1}. The inserted assistant turns are declared `context_only_spans` (the existing direct-trainer
   mechanism, train/direct_sft.py `failed`/`context_only_mask`), so the repeated action is never a target. Only turns
   where a_t is a read whose rerun is side-effect free qualify (the runtime replay says so; effects are data).
3. **Authored recovery worlds (static curriculum, CPU).** A curriculum family through
   `ts-host/scripts/inline-curriculum` (`--authored-plans --supervise-reference-answers`) whose reference plan makes a
   natural first query that returns empty (the relation is stored in the other direction, the first page is empty, a
   filter is too narrow), then changes the query on that observation. Exact oracle at the root. Seeded worlds give
   breadth (relations, pages, filters, schedules); held split by world seed.

## Checks

- Every example's no-progress observation is classified by the exact rules above; the classifier is crisp code with
  unit tests on the loop cases of the dragnet (multihop owns/owned_by, route_replan `world.look()`, late_argmax
  `page_count()`, entailment `facts.page(1)`, frontier `graph.links`).
- The supervised action differs from the no-progress action (exact comparison of the call text after whitespace
  normalisation).
- Held split: source 3 by world seed; sources 1–2 inherit the source rows' split, and train branches never come from a
  document that shares a source group with a held document.
- Measured effect: the in-run task evaluation's exact-repeat counter (Pop's `identical_turn_limit` instrumentation,
  f8f9a99c) on the held packet, before and after a run that includes the family. The family is training data, not a
  model-facing prompt change, so it needs no live A/B before admission; its effect is read from the task cadence.

## Order

Source 1 first (a selection over existing data, adds a sampling group to r4), then source 3 (new breadth, no teacher
cost), then source 2 (needs the runtime replay harness).

## Status 2026-10-13: source 1 (mined natural recoveries)

Pop continuation at 13:36 CEST: DGX published the immutable source-1 selection as
`natlang-recovery-selection-r3-20261013-v1`. Pop restored and verified it, then
bound it in both phases of the shared `neuralese-direct-v3` recipe (`257d8a48`):
`program-header` for direct_sft and `whole-trajectory` for direct_trajectory,
rules `failed`, `empty`, and `identical`, with recovery cohort repeat two. The
selection changes sampling metadata only, retaining original text, masks,
source groups and split. The running Pop QAD v9 process still has its launch
recipe; the new selection takes effect only at a safe recipe handoff. The
first teacher-generated pilot is separately registered as
`luna-recovery-failure-seeded-20261013-v1` (`c1baf3ad`): ten exact accepted
parents and nine approved repair actions, with ten bad seed actions and two
hardcoded teacher actions excluded from targets. DGX has its exact files and
owns twin rendering into the next union. This pilot is deliberately small;
expand across failure mechanisms and task domains before giving teacher
recovery a large sampling share. In particular, the mined `failed` rule
dominates; measure the distinct empty/identical behaviors and successful
task outcomes separately so a large error-fix volume does not hide missing
recovery diversity.

Built: the exact classifier `training/neuralese/natlang_neuralese/train/recovery_mining.py` (rules identical / empty /
failed; the failed form reuses `text_supervision.failed_feedback`, which wraps `FAILED_TOOL_FEEDBACK`), the streaming
scanner `scripts/neuralese_data/mine_recovery.py` (bounded memory, spool then held-group filter), and
`tests/neuralese/test_recovery_mining.py` (the five dragnet loop cases as synthetic observations, scanner and guard).
Exact choices: an action is the assistant message without `<think>`, whitespace-collapsed; an observation drops the
`Stored local` line before comparison; "empty" is only `[]`, `{}`, `null`, `""` (console output or a declaration
status is information, so `console:\npages: 3\nnull` is not empty, it is caught by "identical" on the rerun).
Program-header and full rows supervise their last assistant turn; whole-trajectory rows every assistant turn.

Scan of `natlang-target-union-20261011-r3` (mellum21 twin; the LFM twin has the same documents by construction, ids are
tokenizer-agnostic): program-header 365,784 rows, whole-trajectory 167,371, full 79,391; ledger unit
`mine-recovery-r3`, about 5 minutes on CPU. Manifest and summary:
`/mnt/external/natlang-development-data/recovery/r3-mellum21/{recovery-selection.jsonl,summary.json}` (sha256
`dae1c4567e34c25ac72ff103cc1ab09713e681ba75424b7156f113a0d80bf6c9`; one JSON line per (rendering, document id, turn)).

| | train | held (test, never train data) |
|---|---|---|
| selections (all renderings) | 18,578 | 945 |
| rule failed | 15,989 | 838 |
| rule empty | 2,019 | 87 |
| rule identical | 895 | 26 |
| program-header / whole-trajectory / full | 11,580 / 5,981 / 1,017 | 451 / 253 / 241 |

Distinct train decisions: 13,743 (about 3.8% of the 357,632 train program-header rows); the three renderings overlap on
the same decisions. Rules can co-occur on one selection. Train selections dropped by the held-source-group guard: 0
(r3 already has no mixed groups). The failed rule dominates (error then fix); the progress-after-no-progress signal
the loops lack is the 2,019 empty and 895 identical train selections (about 1,970 distinct, mostly
workflow_invoice-processing 972, s72 645, recovered_clevr 341, commaqa, agent-trace-observability, musique). 103 families.
The natural share is small, as the design expected, so sources 3 and 2 remain needed for breadth of the empty/identical cases.

Registering for union r4 (coordinator decides; not registered here): a `recovery` sampling group over the r4 union
rows, keyed by document id. For program-header and full the row is the document (its supervised last turn is the
selected turn, so the group is the set of those row ids); for whole-trajectory the selected turn is one of several
supervised turns, so the group selects the row and any weight applies to the document. Regenerate the manifest against
the r4 union files (`mine_recovery.py --input RENDERING=PATH`), register it as a corpus entry with the summary sha, and
give the group the same split as its source rows. Optionally restrict the group to the empty/identical selections
(about 2.9k train) and treat the failed ones as already covered by the context-only mechanism.
