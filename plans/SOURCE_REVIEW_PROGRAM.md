# Source review as data plus a natlang reviewer (review items P2 and P10)

Status: implemented on main; the owner reviews after the fact (plans/OWNER_REVIEW.md). Section 10 records what was built
and the decisions taken where this draft left a choice.

Source: plans/NATLANG_NATIVE_REVIEW.md, rows P2 and P10.

- **P2.** `ts-host/src/teacher/source-review.ts:19-440` holds about 400 lines of per-item review verdicts as
  TypeScript literals. New verdicts need a code change, and the reasoning behind them was produced by agents
  outside any program.
- **P10.** The review-packet scripts (`scripts/build_*_review_packet.py`, `scripts/bind_*_receipts.py`) bind exact
  hashes around a semantic review that is done outside the repository and read from an annotation file.

The design: the verdicts become a data file with an exact migration; two natural-language functions recommend at
intake (`reviewSourceItem`) and on produced rows (`reviewSourceRow`); a receipt records which natlang reviewer
(by hash) said what and which explicit decision followed. The exact hold rules stay crisp. Admission stays explicit.

## 1. What the current code is

`ts-host/src/teacher/source-review.ts` (499 lines, sha256 of the file at the review commit
`abb6f5594086b3672b6826a81729ff916b8152a290a29ffd6359c217e3da9a74`):

| Lines | Content |
| --- | --- |
| 1-2 | imports (`sourceContractsFor` from `../benchmarks/registry.js`) |
| 4-17 | type `SourceReview` (dataset, id, aliases, text, annotatedLabel, status `pending` or `resolved`, reason, optional pins `sourceRevision`, `sourceSnapshotSha256`, `sourcePrompt`) |
| 18-25 | `SOURCE_CONTRACT_REVIEWS`: 2 holds of a whole task family on a primary source id (both HotpotQA `knowledge_evidence`) |
| 27-440 | `SOURCE_REVIEWS`: 184 entries, all `pending`, none `resolved` |
| 442-446 | `anliReviewText`: joins four story fields with newlines |
| 448-461 | `pendingSourceReview(dataset, id, record?)` |
| 463-499 | `sourceReviewReason(record)` |

The 184 entries, by dataset: musique 71, tatqa 40, scifact 20, folio 18, banking77 13, anli 7, natlang-inline-curriculum
5, sms_spam 4, sst2 2, hotpotqa 1, `workflowevals:agent-trace-observability` 1, kqapro 1, entailmentbank 1. 41 entries
carry aliases (71 alias strings in all, 255 identity strings with the ids). 12 entries carry pins (11 musique, 1 tatqa).
26 entries contain non-ASCII text. One block is generated: lines 265-288 build seven `anli` entries by `.map` over
tuples. The largest entry is 1,107 bytes of JSON; the whole registry is about 103 KB.

Three kinds of rule are in `sourceReviewReason` and `pendingSourceReview`, besides the records:

- **Dataset knowledge held as code:** the inference from `curriculum.family` to a dataset
  (`entailment_premises` to `entailmentbank`, `kqapro_question` to `kqapro`, `anli_batch` to `anli`, `folio_batch`
  and `folio_entailment` to `folio`; lines 466-467), the display-name alias `HotpotQA distractor` to `hotpotqa`
  (line 471), the hold of the whole `banking77` family `cross_source_folders` (lines 473-478), the lists of datasets
  whose identities also come from `source_groups` (`folio`, `entailmentbank`) or from `curriculum.shape` (`folio`,
  `kqapro`, `entailmentbank`) (lines 492-494).
- **Exact matching:** the hold semantics listed in section 3.
- **Exceptions:** `reviewedIds`, the ids that a reviewed replacement contract covers, which are released
  (`sourceContractsFor(dataset).filter(isReviewedVariant)`, lines 495-496).

Callers (all keep working unchanged):

| Caller | Use |
| --- | --- |
| `ts-host/src/teacher/curriculum-policy.ts:4, 20` | `quarantineReason` returns `source_review_pending` first |
| `ts-host/scripts/inline-curriculum/folder-data.mjs:5, 39` | `pendingSourceReview` at source-row intake (`quarantine(dataset, id, 'source_review_pending')`) |
| `ts-host/scripts/inline-curriculum/workflow-sources.mjs:6, 102` | the same, for `workflowevals:<source>` aliases |
| `ts-host/scripts/inline-curriculum/sources-ai2.mjs:11` | `anliReviewText` |
| `ts-host/scripts/verify-recurrence-pool.mjs:17` | `sourceReviewReason` |
| `scripts/prefer_current_generation_cases.mjs:11, 88` | `sourceReviewReason` as `sourceHeld` |
| `ts-host/test/source-review.test.mjs`, `ts-host/test/folder-families.test.mjs:52-76` | exports and the shape of `SOURCE_REVIEWS` |
| `ts-host/scripts/admission-dispositions.mjs:14-15` | maps the `source_review_pending` reason to category `oracle_or_source_review` (never a DPO negative) |

## 2. Decisions, part by part

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| The 184 verdict records | data | `training/source-reviews/holds.jsonl` | Each verdict is a reviewed fact with a reason, not logic. New verdicts are data additions with a manifest, like other data (`AGENTS.md`: publish data additions with immutable manifests). |
| The 2 family-and-primary-id contract holds | data | `training/source-reviews/family-holds.json` | Same kind of fact; also receives the `banking77` `cross_source_folders` family hold (now code, lines 477-478). |
| Dataset inference from curriculum family, display-name alias, identity-source traits | data | `training/source-reviews/datasets.json` | Knowledge about datasets. Adding a dataset needs no code. |
| `pendingSourceReview`: exact id, alias, dataset, status and pin matching | crisp | `source-review.ts` | Exact verifier; the hold rules stay in code. |
| `sourceReviewReason`: the order of checks and the `source_review_pending` result | crisp | `source-review.ts` | Exact. |
| `anliReviewText` | crisp | `source-review.ts` | Defines a stable identity from visible text. |
| `reviewedIds` exception | crisp | `source-review.ts` | Reads the benchmark registry. |
| Recommend at intake whether an item deserves a hold | fn | `reviewSourceItem` | A judgment about meaning: does the visible text support the label under the family's contract. |
| Recommend on a produced row whether the target is semantically equivalent to the source answer | fn | `reviewSourceRow` | The semantic review that is done by agents outside the repository today (P10). |
| Optional second opinion by a different executor | crisp control over the same fns | `secondOpinion` setting | Independent agreement signal. |
| Compare two recommendations | crisp | `agree(a, b)` | Exact equality of the recommendation enums. |
| Exact pre-checks on rows: exact match, exact source span, numeric normalization | crisp | `row/precheck` | Replaces model work that is exact. |
| Receipt of reviewer hash, recommendation, explicit decision | crisp | `source_review.py receipt` | Hash binding, as in `bind_held_action_review_receipts.py`. |
| The decision (hold, clear, defer) | host (explicit) | `source_review.py decide` | A person or an agent session decides and signs; the program does not. |
| Admission of any row into training | host (explicit) | `training/neuralese_corpora.json` | Unchanged; no receipt grants admission (`training_admission: false` on every receipt). |
| Binding hashes between packets, results, traces and source rows | crisp | `scripts/bind_*_receipts.py`, `build_*_packet.py` | Exact identity checks; these stay. |

## 3. Exact migration of the verdicts to data

### 3.1 Files

```
training/source-reviews/
  holds.jsonl           one SourceReview per line, in the array's order
  family-holds.json     the 2 contract holds, plus the banking77 family hold
  datasets.json         family-to-dataset inference, display-name aliases, identity-source traits
training/corpus-manifests/source-reviews-<date>-v1.json
                        immutable SHA-256 manifest: file hashes, record count, per-dataset counts
```

A `holds.jsonl` line has exactly the current fields, with the optional pins present only when the entry has them:

```
{"dataset": "...", "id": "...", "aliases": ["..."], "text": "...", "annotatedLabel": "...",
 "status": "pending", "reason": "...", "sourceRevision": "...", "sourceSnapshotSha256": "...", "sourcePrompt": "..."}
```

`family-holds.json` entries: `{dataset, family, primarySourceId | null, reason}`. The two current rows keep their
`primarySourceId`; the `banking77` row has `primarySourceId: null`, meaning every task of that family.

### 3.2 Byte-preserved fields

1. Every `dataset`, `id` and alias is copied as the same JavaScript string value (so the same UTF-8 bytes). The
   migration compares the list of identity strings before and after byte for byte.
2. `text`, `annotatedLabel`, `reason`, `sourcePrompt` are the same string values (the source mixes literal characters
   and `\u` escapes; JSON parsing gives the same code points either way).
3. The seven generated `anli` entries are written out as explicit records at the same position in the order, with
   `aliases: []` and `status: "pending"`.
4. Order is preserved: `pendingSourceReview` returns the first match (`Array.find`), so order is part of behavior.

### 3.3 Pre-migration fingerprint

Computed from the extracted records of the source at the review commit (key order `dataset, id, aliases, text,
annotatedLabel, status, reason, sourceRevision, sourceSnapshotSha256, sourcePrompt`, undefined keys omitted, one
`JSON.stringify` of the array):

| Quantity | Value |
| --- | --- |
| Records | 184 |
| Identity strings (ids plus aliases) | 255 |
| sha256 of the records array | `006941286fc79f951bfde45d87edcbd490b88549bd77215172fca9b31743f4e2` |
| sha256 of identities (`dataset NUL id NUL alias...`, joined by newline, in order) | `0d1cd22acda8eeb2c1c1202aa9b6dafaac7772202ba84682358809768ec3af31` |

The implementation recomputes both values from the compiled pre-migration module (`dist/teacher/source-review.js` at
the parent commit) and from the loaded data, and the test asserts both equal.

### 3.4 Loader

`source-review.ts` keeps its exports (`SourceReview`, `SOURCE_REVIEWS`, `SOURCE_CONTRACT_REVIEWS`, `anliReviewText`,
`pendingSourceReview`, `sourceReviewReason`) and their types. `SOURCE_REVIEWS` is read from `holds.jsonl` at module
load by an exact loader that validates each line (known keys, `status` in `pending | resolved`, `aliases` array of
strings, no duplicate `dataset|id` or `dataset|alias`) and fails on any problem. `SOURCE_CONTRACT_REVIEWS` keeps its
current shape (`{dataset, family, primarySourceId, reason}` for the entries that have a primary id). The dataset
tables replace the inline conditionals; the order of checks in `sourceReviewReason` is unchanged.

### 3.5 Hold semantics that stay identical

1. Only entries with `status === 'pending'` hold. A `resolved` entry never holds.
2. Without a record, `pendingSourceReview(dataset, id)` matches the id or any alias within the same dataset
   (ID-only lookups enumerate the pending review, `source-review.ts:451`).
3. With a record, each pin that the entry has must match: `sourceRevision` in `record.source_revisions`,
   `sourceSnapshotSha256` equal to `record.external_source.snapshot_sha256`, and `sourcePrompt` equal to
   `semantics.files[semantics.root]` with `semantics.expected` equal to `annotatedLabel`.
4. The first matching entry wins.
5. `sourceReviewReason(record)` returns `'source_review_pending'` or `undefined`, with this order: no string dataset
   gives `undefined`; the `banking77` family hold; a contract hold with the primary source id as
   `dataset_records[0]`; for `anli`, a pending entry whose `text` equals the joined visible text of any story; then
   a pending review for any of `dataset_records`, `source_ids`, `source_groups` (only for `folio` and
   `entailmentbank`) and `curriculum.shape` (only for `folio`, `kqapro`, `entailmentbank`), unless the id belongs to
   a reviewed replacement contract.
6. `quarantineReason` still calls it first after the `treedst` check (`curriculum-policy.ts:20`).

### 3.6 Migration test (exact equality, not sampling)

At implementation, in `ts-host/test/source-review-migration.test.mjs`:

1. The two fingerprints in 3.3 equal the loaded data's.
2. Deep equality of the new `SOURCE_REVIEWS` with a frozen JSON snapshot of the old module's export, field by field,
   in order, and UTF-8 byte equality of every identity string.
3. A differential test with a frozen copy of the old functions in a test fixture: for every entry and every alias,
   the record shapes of `source-review.test.mjs` (dataset-only, with pins matching and with each pin broken,
   `source_ids` instead of `dataset_records`, wrong dataset, `folio` with `source_groups`, `kqapro` with `shape`,
   `anli` stories, `HotpotQA distractor`, the contract-hold family variants, the banking family) give equal results
   from old and new `pendingSourceReview` and `sourceReviewReason`.
4. The existing `source-review.test.mjs`, `folder-families.test.mjs` and `curriculum-policy` tests pass unchanged.
5. The migration is one commit: data files and manifest added, literals removed from the `.ts`, callers untouched.

A rollback is `git revert` of that commit.

## 4. Where the natural-language reviewer works

### 4.1 `reviewSourceItem` (intake, P2)

Called when a source item enters a family's pool and has no registry entry (`folder-data.mjs:34-41`,
`workflow-sources.mjs:102`, source adapters in `sources-ai2.mjs`). Items that already have an entry are not
reviewed again; their entry is the standing decision.

```
args: item: { dataset: string, id: string, visible: Untrusted<string>, annotated_label: Untrusted<string>,
              contract: string, answer_format: string | null },
      precedents: { id, concern, reason }[]     // up to the most similar held items of the same dataset, chosen by crisp text similarity
returns: ItemRecommendation

type Concern = "none" | "label-disagrees-with-source" | "unstated-premise" | "ambiguous-question"
             | "contract-mismatch" | "needs-context"
type ItemRecommendation = {
  recommendation: "admit" | "hold",
  concern: Concern,
  reason: Is<string, "one to three sentences naming what the visible text states, lacks or contradicts">,
  evidence: { quote: Is<string, "occurs in item.visible"> }[],
  proposed_entry: { reason: string } | null,         // present exactly when recommendation is "hold"
  confidence: "high" | "medium" | "low",
}
```

Steps:

1. Read `contract` (the question or criterion the family asks about this item) and `item.visible`.
2. Write in one phrase the answer that `item.visible` supports under `contract`.
3. Compare it with `annotated_label`. When they agree, go to step 4. When they differ, set concern
   `label-disagrees-with-source`, quote the sentence of `item.visible` that supports your answer, and go to step 6.
4. List the facts that `annotated_label` needs. Mark each as stated in `item.visible` or not. When a needed fact is
   not stated, set concern `unstated-premise` and name the fact.
5. Read the question once more. When two readings give different labels, set concern `ambiguous-question` and
   write both readings. When the label depends on information that `item.visible` does not carry (a sender's
   context, a document outside the item), set concern `needs-context`.
6. When a `precedents` entry has the same concern, use its wording style for `reason`.
7. Return `hold` when a concern was set; otherwise `admit` with concern `none`. For `hold`, return `proposed_entry`
   with the `reason`.

The crisp side checks the quotes against `item.visible` and that `proposed_entry` is present exactly for `hold`.

### 4.2 `reviewSourceRow` (produced rows, P10)

Called on rows a collection or generation run produced, where the packet scripts currently read a separate
annotation file. The model already did the work; this function is the reviewer that writes the annotation.

```
args: row: { dataset: string, split: string, source_groups: string[],
             question: Untrusted<string>, answer_format: string,
             evidence: Untrusted<string>,              // the source passage or record
             gold: Untrusted<string>, actual: Untrusted<string> },
      precheck: { exact_match: boolean, actual_is_exact_source_span: boolean, numerically_equal: boolean | null }
returns: RowVerdict

type RowStatus = "equivalent" | "normalization-candidate" | "mismatch" | "ambiguous"
type RowVerdict = {
  status: RowStatus,
  rationale: Is<string, "one or two sentences citing the evidence for the status">,
  evidence: { quote: Is<string, "occurs in row.evidence"> }[],
  confidence: "high" | "medium" | "low",
}
```

Steps (called only when `precheck.exact_match` is false; an exact match is `equivalent` without a call):

1. Read `question`, `answer_format` and `evidence`.
2. Decide whether `actual` answers `question` under `answer_format`, using `evidence`. Quote the sentence of
   `evidence` that carries the answer.
3. Compare the meaning of `actual` and `gold`. Same meaning with different wording or extra qualifiers that
   `evidence` supports: `equivalent`. Same meaning but a different representation that a fixed rule could normalize
   (units, thousands separators, case): `normalization-candidate`.
4. Different meaning: `mismatch`. When `question` allows both answers under `evidence`: `ambiguous`.
5. Write `rationale` in one or two sentences.

Existing categories of the v7 review (`plans/neuralese/S1_QUESTION_COLLECTION_V7_SEMANTIC_REVIEW_2026-10-08.json`)
map as: `receipts` and `semantic_normalization_candidate` to `equivalent` and `normalization-candidate`,
`strict_semantic_mismatches` to `mismatch`, `semantic_ambiguities` to `ambiguous`, `held_numeric_representation` to
the crisp `numerically_equal` precheck, `transport_error` stays crisp (no model verdict).

### 4.3 Optional second reviewer

A setting `secondOpinion` names a second executor identity (a different model or prompt variant). The program calls
the same function with it and records both recommendations. `agree` compares the enums
(`recommendation` for items, `status` for rows). Disagreement appears in the intake report as `needs-human` and is
not resolved by either side. Costs one call per item; used for samples first (see section 7).

## 5. Receipt and decision

Recommendations never change the registry or any admission. A recommendation and the decision that follows are
recorded together:

```
{
  "schema": "natlang.source-review-receipt/1",
  "kind": "item" | "row",
  "subject": { "dataset": "...", "id": "...", "subject_sha256": "<canonical JSON of the reviewed input>" },
  "reviewer": { "kind": "natlang", "function": "reviewSourceItem", "definition_key": "<call-store definition key>",
                "reviewer_hash": "natlang@<first 16 hex of sha256(definition source + compiler version + executor identity)>",
                "call_id": "<call-store id>" },
  "second_reviewer": { ... same shape ... } | null,
  "recommendation": { ...ItemRecommendation or RowVerdict... },
  "agreement": "agree" | "disagree" | "single",
  "decision": null | { "value": "hold" | "clear" | "defer", "by": "owner" | "agent:<session>", "at": "<ISO time>", "note": "..." },
  "registry": { "before_sha256": "...", "after_sha256": "..." },
  "training_admission": false
}
```

- `reviewer_hash` identifies the exact natlang reviewer: the function source revision, compiler version and executor
  identity. The call store already keys definitions this way (`definition_key`, `applications/specializer/main.ts:35`).
- The decision is made by `python3 scripts/source_review.py decide <receipt id> --decision hold|clear|defer`, run by
  the owner or by an agent session that signs with its name. A `hold` decision appends the receipt's
  `proposed_entry` to `holds.jsonl` as a `pending` record and writes a new manifest; `clear` writes the receipt
  only; `defer` leaves the receipt open.
- `training_admission` is always false. Admission, split and quality decisions stay explicit in
  `training/neuralese_corpora.json` (`AGENTS.md`). A `clear` decision does not admit anything; it records that the
  item was reviewed.
- `scripts/bind_held_action_review_receipts.py` keeps binding exact hashes. Its inputs gain the receipts of
  `reviewSourceRow` where today they read an annotation file; the packet builders
  (`build_step5_native_action_review_packet.py:92-94` reads `annotations.rows[...].disposition` in `candidate | hold`)
  accept the receipt's decision as that disposition, and still refuse a row without an explicit one.

## 6. Refinement-type candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `ItemRecommendation.evidence[].quote`, `RowVerdict.evidence[].quote` | `Is<string, "occurs in item.visible">`, `"occurs in row.evidence"` | crisp |
| `ItemRecommendation.proposed_entry` | `Is<Entry \| null, "present exactly when recommendation is hold">` | crisp |
| `ItemRecommendation.concern` | `Is<Concern, "none exactly when recommendation is admit">` | crisp |
| `ItemRecommendation.reason`, `RowVerdict.rationale` | `Is<string, "one to three sentences naming what the visible text states, lacks or contradicts">` | judged |
| `RowVerdict.status` | `Is<RowStatus, "equivalent whenever precheck.exact_match is true">` | crisp |
| `item.visible`, `item.annotated_label`, `row.evidence`, `row.actual`, `row.gold` | `Untrusted<string>` | crisp marking |
| `precedents[].id` | `Is<string, "the id of a record in holds.jsonl of the same dataset">` | crisp |
| `holds.jsonl` line | `Is<SourceReview, "known keys only; status pending or resolved; no duplicate identity">` | crisp (loader) |
| `Receipt.decision.by` | `Is<string, "owner or agent: followed by a session name">` | crisp |
| `Receipt.reviewer.reviewer_hash` | `Is<string, "equals the hash recomputed from the stored definition, compiler and executor identity">` | crisp |

## 7. Model-facing changes needing live measurement

Both functions are new. Measurement uses the registry itself as labelled data:

1. **`reviewSourceItem` against the 184 existing holds plus a sample of admitted items from the same datasets.**
   Metrics: recall of the existing holds (recommendation `hold`), false-hold rate on the sample, concern agreement with
   the recorded reason text, quote validity. About 48 samples per variant on the held-out part (hold out the
   datasets or time-split the newest entries).
2. **`reviewSourceRow` against the 13 receipts, 3 strict mismatches, 2 ambiguities and the normalization
   candidates of the v7 review**, plus exact matches as controls.
3. **Second reviewer agreement** with the first on the same samples.
4. **`Untrusted<string>` rendering** of item and row text. Add these functions to the existing unmeasured entry in
   `plans/MODEL_FACING_CHANGES.md`.
5. **Variant:** the cause table in step 3-5 spelled as steps against the `Concern` type text only.

No existing model-facing text changes: the registry move is not seen by any model.

## 8. Order of work, when approved

1. Migration commit (section 3), which needs no model and no owner decision beyond approval of the layout.
2. `source_review.py` (receipts, decide) and the manifest writer.
3. `reviewSourceItem`, run in a teacher window on the sample (section 7); shadow only: recommendations are written
   to the intake report and not attached to decisions.
4. `reviewSourceRow` replacing the annotation-file authoring for the next packet.
5. A second reviewer on a sample.

## 9. Questions for the owner

1. Should an item with a `hold` recommendation and no decision yet be paused from generation until a decision is
   recorded (more conservative), or flow on (this draft: no behavior change until a decision creates a pending entry)?
2. Where should the registry live: `training/source-reviews/` (proposed, beside other training data), or inside
   `ts-host/` (next to the code that reads it, but the data is training-side knowledge)?
3. Who may sign a `decide`: the owner only, or any agent session? Receipts record the signer either way.
4. Should `resolved` entries (none today) carry a `resolution` object (what replaced the item, the new identity)
   so a release is as traceable as a hold? The matching rules stay the same.
5. The second reviewer's executor: a second model (needs another endpoint) or the same model with a different prompt
   variant?

## 10. Implementation notes (2026-10-09)

What was built, in the order of section 8 (steps 1, 2 and the function definitions of 3 and 4; the live measurements of
sections 7 and 8 wait for a teacher window):

1. **Migration (one revertable commit).** `training/source-reviews/{holds.jsonl,family-holds.json,datasets.json}` and
   `training/corpus-manifests/source-reviews-20261009-v1.json`; `source-review.ts` loads and validates them. The pinned
   fingerprints of section 3.3 were recomputed from the compiled pre-migration module and are correct: 184 records,
   255 identity strings, records sha256 `0069412...f4e2`, identities sha256 `0d1cd22a...af31`, source file sha256
   `abb6f559...a9a74`. `ts-host/test/source-review-migration.test.mjs` asserts them, deep equality with a frozen export
   of the old module, and a differential test against a frozen copy of the old functions (over 5,000 record shapes plus
   odd inputs). The banking77 family hold is the third row of `family-holds.json` (`primarySourceId: null`); its reason
   text is new (the old code had a comment, not a reason string) and does not affect matching.
   The manifest is a registry manifest (`natlang.source-review-registry-manifest/1`), not a training corpus: it is not
   listed in `training/neuralese_corpora.json` and grants no admission.
2. **Receipts and decisions.** `scripts/source_review.py` (`receipt`, `decide`, `manifest`, `show`). A receipt carries
   the reviewer as `natlang@<16 hex>` over the sha256 of the definition source, the compiler version and the executor
   identity (recomputed on receipt), the canonical hash of the reviewed input, the recommendation, `decision: null` and
   `training_admission: false`. `decide` needs `owner` or `agent:<session>`; only a `hold` on an item receipt appends a
   pending record to `holds.jsonl` and writes the next manifest; `clear` and `defer` never touch the registry; a decided
   receipt cannot be decided again.
3. **Reviewers.** `reviewSourceItem` and `reviewSourceRow` are built-ins (`ts-host/src/builtin/*.nl`) wrapped by
   `ts-host/src/teacher/source-review-nl.ts`: exact prechecks, precedent choice by word overlap, the crisp checks of the
   typed contract (quotes occur in the text, `proposed_entry` exactly for a hold, `equivalent` on an exact match),
   `pluggable` modes `crisp | nl | shadow` with default `crisp` (recommends nothing), and an optional second executor.
   An exact match is `equivalent` from a crisp reviewer without a model call.
4. **Packet scripts.** `build_step5_native_action_review_packet.py --row-reviews` reads the decided row receipts where
   `--annotations` was read (the decision hold or clear becomes the disposition hold or candidate; a row with no explicit
   decision is refused; the assessment records the v7 category). `bind_held_action_review_receipts.py --row-reviews`
   requires an explicit receipt per selected item over the item's exact gold and actual values and records it in the
   binding. Both keep the old path when the option is not given.

Decisions on the open questions of section 9:

1. **Generation pause.** Today's behaviour is kept: a `hold` recommendation without a decision does not pause
   generation; only a recorded `hold` decision creates a pending entry that holds.
2. **Registry location.** `training/source-reviews/`, as proposed. `NATLANG_SOURCE_REVIEWS` overrides the folder.
3. **Who signs.** `decide` accepts `owner` or any `agent:<session>`; the receipt records the signer.
4. **`resolved` entries.** Unchanged (no `resolution` object yet); the loader accepts the status and the matching rules
   are the same.
5. **Second reviewer.** `ReviewOptions.second` takes any executor identity and call function; which model it is remains
   a deployment choice.
