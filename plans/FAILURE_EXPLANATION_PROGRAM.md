# Failure explanation as advisory natlang programs (review items P5 and P11)

Status: draft for owner review (plans/OWNER_REVIEW.md). Nothing is built until the owner has reviewed it. No code
changes accompany this document.

Source: plans/NATLANG_NATIVE_REVIEW.md, rows P5 and P11.

- **P5.** Failure and rejection categories are assigned by substring and regular expression; what matches nothing
  falls to "needs review". The two sites are `scripts/audit_student_projection_failures.py:9-37` and
  `ts-host/scripts/admission-dispositions.mjs:5-28`.
- **P11.** When an eval gate fails (`training/neuralese/natlang_neuralese/eval/foundation.py`, `self_feedback.py`,
  `train/foundation_schedule.py`), the explanation is written by hand.

The design: three advisory natural-language functions, applied only where the crisp classifiers give up.

- `explainFailure`: one student-projection failure that the audit left `unclassified_needs_review`.
- `triageRejections`: the admission reasons that fall to `unclassified_review_pending`.
- `explainGateFailure`: a failed gate report.

**Gates never change.** No function here decides anything the pipeline acts on. Every output is written beside the
crisp result, labelled as advisory with the explainer's hash, and no decision, count, admission, DPO hold, training
row or threshold reads it.

## 1. What the current code is

### 1.1 `scripts/audit_student_projection_failures.py`

`classify(candidate)` (lines 9-35) returns a sorted tag list from crisp tests over a receipt:

| Tag | Test | Line |
| --- | --- | --- |
| `infrastructure_or_replay` | error message contains `prefix observation` or `student_score` | 11 |
| `generation_truncated` | error contains `incomplete_projection` | 12 |
| `request_budget` | error contains `request_budget` | 13 |
| `api_or_type_contract` | tool feedback contains one of `typeerror:`, `referenceerror:`, `no such`, `not a function`, `unknown-field`, `type-mismatch`, `not-writable`, `expected a sentence` | 25 |
| `state_recovery` | feedback contains `nothing else from this eval was kept`, `cannot move an entry into itself`, `file not found:` | 26 |
| `effects_or_file_state` | `checks.files`, `checks.effects` or `checks.file_return_consistency` is false | 27 |
| `honest_stop` | `checks.honest_stop` is false | 28 |
| `finish_protocol` | feedback contains `returns no value`, `before returning a value`, or `omit value` with `reason` | 29 |
| `incomplete_execution` | `budget exhausted` in the outcome detail, or `checks.expected_status` false | 30 |
| `repeated_action` | the same action JSON occurs twice | 31 |
| `delegation_review` | more than one `nl` delegation in eval code (flagged, "not itself a fault") | 32 |
| `answer_mismatch_needs_review` | `checks.answer` false | 33 |
| `unclassified_needs_review` | none of the above | 34 |

`audit(root)` (lines 37-56) aggregates counts per tag and family and writes a report with
`heuristic_labels_require_review: true`, `accuracy_estimate: null`, `too_hard_labels_created: 0`,
`training_rows_created: 0`, `preference_labels_created: 0`. The report is created with mode `x` (never overwritten).

### 1.2 `ts-host/scripts/admission-dispositions.mjs`

`RULES` (lines 5-20) is an ordered list of seven rules, each `{category, action, matches(reason)}`:
`migration_or_replay_pending` (two rules), `duplicate_or_superseded`, `evaluation_or_unsupported` (two rules),
`oracle_or_source_review`, `candidate_failure`. `classifyAdmissionReason` (22-28) tries each rule on the reason and
on its prefix before the first colon; no match gives `unclassified_review_pending` with next action
`retain_raw_evidence_and_review_before_training`. `dpoHoldReasons` (35-39) holds five categories, including
`unclassified_review_pending`, from preference pairs. Callers: `build-preference-pairs.mjs:21, 103, 113`,
`inline-curriculum/corrections.mjs:19, 74, 79`, `snapshot-generated-training.mjs:13, 63`.

### 1.3 Gates

| Gate | Where | Pass rule | What is written when it fails |
| --- | --- | --- | --- |
| Token-aligned identity control | `eval/foundation.py:119-135` | three exact zeros per row: `raw_transport_max_abs`, `identity_readback_logits_max_abs`, `full_output_reference_max_abs` | the report (`token_aligned_reference_passed: false`, `learned_channel_qualified: false`, per-row numbers) is written to a fresh immutable output, then `SystemExit('token-aligned identity reference failed')` |
| Feedback gate of the foundation schedule | `train/foundation_schedule.py:50-51, 242` | per stratum, `agreement >= --agreement-gate` (0.9) and `kl <= --kl-gate` (0.25) | `feedback_gate_passed` in the stage record |
| Self-feedback proposed gate | `eval/self_feedback.py:138-153, 211-225` | per stratum (`first`, `last`, `length_band:*`), `argmax_agreement >= min`, `kl_plain_to_projected_nats <= max`, `quality_ce_gap <= max` (token-weighted means) | `proposed_gate_passed`, per-stratum `passed`, per-window `proposed_gate_passed` |

`AGENTS.md`: "Finishing a warm-up's step count is not qualification. Preserve and diagnose failed gates."

## 2. Decisions, part by part

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Tags that the receipts decide exactly (checks false, budget, error codes) | crisp | `classify` | Exact tests over structured fields. They stay. |
| Tags decided by substrings in tool feedback text | crisp, kept | `classify` | Cheap, deterministic, and measured against the explanations (section 6). They are not removed in this draft. |
| The unclassified bucket | fn | `explainFailure` | The residue is where substring tests have no answer. |
| Compact failure card for one candidate: error, checks, first and last actions, tool feedback excerpts | crisp | `failureCard` | Pure extraction, the same fields `classify` reads, bounded. |
| Proposed new tag when no existing tag fits | fn output, human decision | `FailureExplanation.proposed_tag` | A new crisp tag is added by a person after review, as a code change with a test. |
| Rules for admission reasons | crisp, kept | `RULES`, `classifyAdmissionReason` | The category decides DPO holds and routing; it stays exact. |
| Distinct reasons that classify as unclassified, with counts and example rows | crisp | `unclassifiedBucket` | Grouping. |
| Proposed category, action and matching prefix for each unclassified reason | fn | `triageRejections` | A judgment about what a reason means, offered as a proposed rule. |
| Check a proposed rule: matches its examples and reclassifies no reason that already has a category | crisp | `checkProposedRule` | Exact verifier of the stage output. |
| Accepting a rule | host (explicit) | a commit adding the rule (or a data entry, question 3) | Process of record; the model does not edit rules. |
| Failed gate facts: which strata and metrics failed, observed against limit, margin, which windows | crisp | `gateFacts` | Arithmetic over the report; the model reads numbers. |
| Pattern across strata and windows, candidate causes, next checks | fn | `explainGateFailure` | Interpretation of a failure. |
| The gate's decision and thresholds | crisp, unchanged | `foundation.py`, `self_feedback.py`, `foundation_schedule.py` | Exact; the explanation reads them and never writes them. |
| Writing explanations | crisp | `writeAdvisory` | New files next to the crisp result; fresh, immutable, never merged into counts or gate fields. |

## 3. The natural-language functions

All inputs that originate in logs, tool output, receipts or reports are `Untrusted<string>`.

### 3.1 `explainFailure`

```
args: card: FailureCard, tags: string[]    // the vocabulary of existing tags, passed as data
returns: FailureExplanation

type FailureCard = {
  family: string, program_id: string,
  error: Untrusted<string>,
  checks: Record<string, boolean | null>,        // the receipt's outcome.checks
  outcome_detail: Untrusted<string>,
  actions: Untrusted<string>[],                   // the model's calls, first and last turns, each bounded
  feedback: Untrusted<string>[],                  // the tool feedback after each action, bounded
  turn_count: number, delegation_count: number,
}
type FailureExplanation = {
  tag: Is<string, "one of tags, or the word new">,
  proposed_tag: Is<string | null, "a lowercase snake_case name, present exactly when tag is new">,
  why: Is<string, "one to three sentences naming the action or feedback line that shows the failure">,
  evidence: { quote: Is<string, "occurs in the card"> }[],
  confidence: "high" | "medium" | "low",
}
```

Steps:

1. Read `card.checks` and name the checks that are false.
2. Read the last three `feedback` entries and the last three `actions`; find the first feedback entry that reports a
   problem (an error, a refusal, an empty or surprising result).
3. Name what the program tried at that point and what the tool answered, in one sentence.
4. Compare with `tags`. When one tag describes the problem, return it. When none does, return `new` and write
   `proposed_tag` as a short snake_case name for the problem class.
5. Quote the feedback or action text that supports the choice (`evidence`), and set `confidence` high when a quoted
   line names the problem, medium when it follows from the false checks, low otherwise.

### 3.2 `triageRejections`

```
args: bucket: { reason: Untrusted<string>, count: number, examples: { id, family }[] }[],
      categories: { category: string, action: string, example_reasons: string[] }[]    // existing rule categories, as data
returns: { proposals: RuleProposal[] }

type RuleProposal = {
  reason: Is<string, "a reason in bucket">,
  category: Is<string, "one of categories">,
  next_action: string,
  match_prefix: Is<string, "a prefix of reason that is shared by the reasons it covers">,
  rationale: Is<string, "one sentence comparing the reason with the example reasons of the chosen category">,
}
```

Steps:

1. Read each `bucket` reason and its examples. Reasons often carry a prefix before a colon; the prefix names the class.
2. For each reason, compare it with the `example_reasons` of every category and pick the category whose reasons
   it resembles in meaning (for example "defect not repaired" resembles `candidate_failure`).
3. Take the matching action from that category. When no category resembles the reason, return the category
   `unclassified_review_pending` with the existing action, and say so in `rationale`.
4. Choose `match_prefix`: the longest prefix that covers this reason and its siblings and nothing the categories'
   examples already cover.

`checkProposedRule` (crisp) accepts a proposal only when `match_prefix` matches its reason and examples, and when
running it over every reason the classifier has ever recorded changes no existing non-unclassified result.

### 3.3 `explainGateFailure`

```
args: facts: GateFacts, report: Untrusted<string>   // the report JSON, bounded; facts are crisp
returns: GateExplanation

type GateFacts = {
  schema: string, passed: boolean,
  failed: { stratum: string, metric: string, observed: number, limit: number, direction: "min" | "max", margin: number }[],
  passed_strata: string[], worst_windows: { id: string, metric: string, observed: number }[],
  context: { run: string, checkpoint: string | null, step: number | null, earlier_reports: { path, passed }[] },
}
type GateExplanation = {
  pattern: Is<string, "one to three sentences on where the failures concentrate">,
  candidate_causes: { cause: string, support: Is<string, "cites a field of facts or report">, confidence }[],
  next_checks: Is<string[], "each names a measurement or file that would confirm or exclude a cause">,
  gate_unchanged: true,
}
```

Steps:

1. Read `facts.failed`. Group the entries by metric and by stratum; say which metric fails everywhere and which
   fails only in some strata (for example only `length_band:long`, or only the `last` stratum).
2. Read `facts.worst_windows`; say whether a few windows carry the failure or the failure is spread.
3. Compare with `facts.context.earlier_reports`: say whether the same gate passed before for this run and which
   metric moved.
4. For each pattern write a candidate cause, citing the report field that supports it (a margin, a stratum, a
   window id). Keep causes that the fields can distinguish.
5. For each cause write the next check: a measurement or file that would confirm or exclude it (a specific
   ablation, a held-out split, a log).
6. State that the gate's thresholds and decision are unchanged (`gate_unchanged: true`, a constant the crisp code
   asserts).

The crisp side computes `facts` from the report with the same arithmetic as the gate; it also verifies that
`facts.passed` equals the report's own pass field and fails loudly when they differ.

## 4. Output, labelling, and what never reads it

- **Files.** Next to the crisp output, never inside it:
  - `audit-projection-failures.advisory.json` beside the `--out` report: per candidate id, a `FailureExplanation`
    and the explainer's hash;
  - `<snapshot>.rejection-proposals.json` beside the snapshot: `RuleProposal[]`;
  - `<gate-report>.explanation.json` beside each gate report, created with mode `x` like the reports themselves.
- **Label.** Every file carries `{ "advisory": true, "explainer": "natlang@<hash>", "function": "...",
  "inputs_sha256": "..." }`, where the hash is the call store's definition key plus executor identity (the same
  `reviewer_hash` convention as plans/SOURCE_REVIEW_PROGRAM.md).
- **Never read by:** `classify` counts, `audit()` `counts` and `families`, `classifyAdmissionReason`,
  `dpoHoldReasons`, any `admitRow`, any gate (`foundation.py` `passed`, `feedback_gate_passed`,
  `proposed_gate_passed`), `training_admission`, or the training mix. The existing report guards stay:
  `heuristic_labels_require_review: true`, `accuracy_estimate: null`, `too_hard_labels_created: 0`,
  `training_rows_created: 0`, `preference_labels_created: 0`.
- **Where it runs.** On demand, after the crisp tool finishes (a `--explain` flag for the audit and the snapshot
  tool, `python3 scripts/explain_gate.py <report>` for gates), through the ledger when it needs an executor. It
  runs in a teacher window for the first measurement.

## 5. Refinement-type candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `FailureExplanation.tag` | `Is<string, "one of tags, or the word new">` | crisp |
| `FailureExplanation.proposed_tag` | `Is<string \| null, "snake_case, present exactly when tag is new">` | crisp |
| `FailureExplanation.evidence[].quote` | `Is<string, "occurs in the card">` | crisp |
| `FailureExplanation.why` | `Is<string, "one to three sentences naming the action or feedback line that shows the failure">` | judged |
| `RuleProposal.reason` | `Is<string, "a reason in bucket">` | crisp |
| `RuleProposal.category` | `Is<string, "one of categories">` | crisp |
| `RuleProposal.match_prefix` | `Is<string, "a prefix of reason that no already-classified reason shares">` | crisp (`checkProposedRule`) |
| `GateExplanation.candidate_causes[].support` | `Is<string, "cites a field of facts or report">` | crisp for the field name; judged for relevance |
| `GateExplanation.next_checks` | `Is<string[], "each names a measurement or file that would confirm or exclude a cause">` | judged |
| `GateExplanation.gate_unchanged` | the constant `true` | crisp |
| `GateFacts.passed` | `Is<boolean, "equals the report's own pass field">` | crisp |
| `FailureCard.*` text, `bucket[].reason`, `report` | `Untrusted<string>` | crisp marking |

## 6. Model-facing changes needing live measurement

All three functions are new. Measurement:

1. **`explainFailure` against the substring tags.** Run it on candidates that `classify` tagged (not only the
   unclassified ones) with the tags hidden; measure agreement with `classify`. A high agreement on tagged
   candidates is the evidence that its judgment on the residue is meaningful. For the unclassified bucket itself,
   a hand review of a sample (the owner or an agent signs) gives precision of `tag` and usefulness of `new`
   proposals. About 48 samples per variant.
2. **`triageRejections` against the existing rules.** Hide the rules, run the function on reasons that the rules
   classify, and measure whether its category and prefix agree. Then run it on the real unclassified bucket and
   count proposals that pass `checkProposedRule`.
3. **`explainGateFailure` on recorded failed gates.** The failed reports in `runs/` and `plans/neuralese/` (the
   foundation-control reports, the self-feedback reports with `proposed_gate_passed: false`) are the fixtures. A
   reviewer scores whether the pattern matches the failure and whether the next checks are the ones the
   investigation actually used.
4. **`Untrusted<string>` rendering** of tool feedback, reasons and reports. Add the three functions to the
   existing unmeasured entry in `plans/MODEL_FACING_CHANGES.md`.
5. **Variant:** the existing tag vocabulary passed as a list of names against passed as names with one example line
   each.

No existing model-facing text changes.

## 7. Order of work, when approved

1. `failureCard`, `unclassifiedBucket`, `gateFacts`, `checkProposedRule`, `writeAdvisory`: crisp, testable without a
   model, with unit tests on recorded receipts and reports.
2. The three `.nl` functions, measured in a teacher window (section 6).
3. `--explain` flags and `scripts/explain_gate.py`.
4. A person adds crisp tags or rules from accepted proposals as ordinary code changes with tests.

## 8. Questions for the owner

1. Should the explainer also run on `answer_mismatch_needs_review` (tag 33, `checks.answer` false), which the audit
   already marks as needing review? This draft applies it to `unclassified_needs_review` only.
2. `delegation_review` is flagged "not itself a fault" in the audit (line 32). Should `explainFailure` also judge
   redundant versus necessary delegations? Not included here.
3. `RULES` in `admission-dispositions.mjs` could become a data table of prefixes and categories, so accepted proposals
   are data additions with a manifest instead of code edits (the same move as the source reviews). Wanted?
4. Gate explanations for the Pop-proposed self-feedback gate thresholds (`.99` argmax, `.02` KL, `.05` own-output CE
   gap, awaiting review in `plans/OWNER_REVIEW.md`): the function reads whatever thresholds the report records, so
   it works with the thresholds that bind. Confirm that nothing here should be tuned to them.
5. Where should `explanation` files live for runs under `runs/`: beside the reports (proposed), or in one
   `runs/explanations/` tree?
