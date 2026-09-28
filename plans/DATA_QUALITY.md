# Data quality decisions — 2026-09-28

This continues the complete raw-result audit in [REJECTION_REVIEW.md](REJECTION_REVIEW.md).
Raw histories and old source shards remain evidence. Admission is recomputed; neither an old acceptance nor a
nearby answer is automatically promoted into current training data.

## Were the slow cases stuck?

| Case | Fresh replies / completion tokens | Finding |
| --- | --- | --- |
| Entailment handoff 65 | 13 / 11,208 | Selection churn: permutations of one chain, all rejected. The negative variant's checker has empty GOLD and can never accept. The delegated instruction assumes support exists. |
| FOLIO handoff 142 | 12 / 12,029 | Active finite model checking, including repairs of boolean/number comparisons and predicate encodings. Did not finish the entire batch in ten minutes. |
| FOLIO handoff 217 | 13 / 12,150 | Several children reached results; other children repaired syntax and assumptions. Active work with substantial rework, rather than a frozen server request. |
| CUAD folder 84 | 24 / 8,272 | Execution/context stall. Only one per-file judgment; malformed map callbacks, the former delegation restriction and failed compaction prevented the report. |
| TextWorld retry 6 | 15 / 11,051 | Exploration followed by state confusion. The model's step returned the pre-action room snapshot; later calls compared it against fresh instructions/live state. No completed result. |

Counts exclude replayed prefixes when the journal retains a raw model response. They measure completed responses,
not semantic progress. Evidence and classifications are in
`runs/rejection-audit-20260928/data-quality-v2/slow-cases.json`.
Two relational probes finish correctly in 171.5 and 205.9 seconds. Handoff 589 also completes correctly after
427 seconds. Thus neither elapsed time nor many turns alone identifies a stuck case.

The supervisor now records activity every 30 seconds: saved turns, fresh replies, completion tokens and repeated
identical action sets. Repetition is diagnostic, not an automatic semantic-failure verdict. Five minutes without a
saved reply produces **inactivity_timeout**, distinct from the hard case deadline. Neither produces a negative
training example. The cutoff can still defer a very long individual response; raw partials remain available.

## Contracts and admission

| Question | Current decision |
| --- | --- |
| Does a rewrite improve the text? | Word overlap cannot establish this. Exact reference/alternate rewrites are supported by dataset gold. Other wordings, including an unchanged draft, need a separately configured rubric judge to verify the requested edit and preservation of meaning/facts. No judge means pending review. |
| Must every CoEdIT draft change? | No. Low token overlap did not establish that an edit was needed. Tasks now say review/edit as needed. The returned count is the actual number of changed files. Reference targets remain examples, with alternatives retained. |
| Can negatives hide missed clauses? | No. CSV checks measure positive recall and precision separately, alongside overall row agreement. Malformed quoting, duplicate/empty ids, extra cells, header mismatch and source-file corruption fail the contract. |
| Is an extracted quotation genuine and relevant? | It must occur in the corresponding original contract. An annotated span/alternate can verify the wording; another quoted extent needs an independent semantic verdict. Token overlap alone cannot distinguish non-compete from non-solicitation. Grading uses a bounded source view for long contracts and can defer when omitted context matters. |
| Is the returned summary truthful? | CSV positive counts and INDEX counts must exactly describe the actual report, independently of agreement against noisy gold. INDEX totals must equal the number of classified files. Count-line garbage, negatives and duplicate labels are rejected. |
| Do file moves double-count deleted source paths? | No. Move tasks score one item per original file. Lost/corrupted/duplicated files and unexpected new files fail; label agreement remains an evaluation measure. |
| Should partial benchmark correctness become training examples? | No automatic promotion. Admission holds cases with failed file items or less-than-perfect aggregate answer agreement for review, even when an evaluation tolerance passes. This avoids teaching known mismatched child judgments through an accepted parent. |
| Must a model reproduce the reference delegation layout? | No. Direct, delegated and mixed solutions can be admitted. Reference child evidence differences are notes. Source-specific answer/quotation evidence and declared follow-up causal checks remain enforced. |
| Must it use iterateOn, a named helper, regex-free code or a particular history-reading style? | Those differences are coverage/efficiency notes, not proof of an incorrect result. Actual result/effect/file contracts, required repairs and forbidden unwarranted function edits remain checks. Technique-only preference pairs are retired. The standard SFT builder admits correct direct and delegated runs. |
| Must Hotpot read exactly two article leads? | No. Source annotations now retain the supporting sentences. A correct answer needs an observed annotated supporting sentence, including via a search result or another call; the two arbitrary lead-prefix markers are removed. Reference articles are retained for provenance. |
| Are nearby answer strings automatically equivalent? | No. QA normalization ignores articles and checks numeric consistency. Equivalent units or location qualifiers below the score threshold need a source-aware rubric verdict. Ambiguous day/date questions explicitly request a calendar date when the gold is one. “Friday” against a date in the old wording is not a reliable negative example. |
| Is “premise removed” a proved negative? | EntailmentBank contains an annotated proof, not an exhaustive verifier for all natural-language science proofs. Its removed-premise variants are quarantined and no longer generated. Supported variants remain; the declared verifier certifies a reference premise selection. ProofWriter separately filters on its enumerated single-proof metadata. |

A verdict with missing or insufficient judging evidence is **quality_pending**, not a wrong-answer label. It is
excluded from SFT and from failed-action/result handoffs and preference replay. The judge can explicitly return
`needs_review`. Teacher and judge must have distinct model IDs; grading shares the case request/concurrency/KV
budgets. Judge identity and endpoint hash participate in reuse provenance. No independent judge has been started
in this session; pending free rewrites and alternate clause extents remain excluded.

## Source and compatibility changes

- Contradictory labels on identical labeled inputs are quarantined rather than taking the last row.
- CoEdIT identity uses the visible draft across instruction wording and edit tasks, with multiple targets retained.
- Hotpot identity uses the visible question; duplicate conflicting answers and missing supporting sentences are
  quarantined. The replacement build found one conflicting-answer group.
- CUAD keeps multiple annotated spans. Unusable positive spans, a missing positive annotation or conflicting
  impossible/positive annotations cannot turn into a negative record.
- Generator reports include source quarantine identities/reasons.
- Admission version is `natlang.inline_curriculum_admission/2`; collection provenance has
  `data_quality_version: 2`. Reuse cannot silently cross this policy or judge configurations.
- Legacy folder move/count/rewrite/extraction/article contracts are held out. Old prepared SFT/pair artifacts must
  be rebuilt through current admission before training; they are not retroactively certified.

These are breaking data-policy changes. They deliberately reduce immediately eligible data where correctness is
unestablished. The raw-result audit and review reasons preserve what needs independent adjudication instead of
relabeling it as a model error.

## Generation and validation

Seven replacement cases spanning six folder families pass reference replay. The Bonsai queue adds six distinct
replacement tasks and two measured FOLIO retries. It removes the unsupported premise probe and the legacy CUAD
case; the complete exclusions ledger is `runs/bonsai-recovery/queue-v4.exclusions.jsonl`.

Bonsai uses frozen `runtime-v9`, `queue-v6.jsonl` and the existing append-only journal. Fresh jobs use distinct paths
so old rows and partials are preserved. Default limits remain one case at a time, two model requests in flight,
128 requests and 600 seconds. Multi-item FOLIO/ANLI gets 1,200 seconds; all-file tasks get a size-based deadline up
to 3,600 seconds and up to 512 requests. All retain the inactivity guard and 20 turns per call. The user's three
ad hoc layers per existing file root remain unchanged. The user's later authorization starts one Luna repair worker;
no model training has started.

### Fresh run checkpoint and repairs

- SMS move: all 39 files moved in 369 seconds; eight destination labels disagree with gold. Returned 39 is correct.
  Failed move-item identifiers name original sources, not current locations. No partial benchmark row is admitted.
- SMS INDEX: deadline after 670 seconds, 50 fresh replies. Offline replay of the exact-request journal shows that
  the first 22-item batch completed, and the parent repeated it despite having its labels/counts. This is redundant
  work, not a frozen server. Folder guidance now makes delegation optional and asks models to reuse completed
  judgments and investigate individual disagreements. Identical sibling answers are not themselves stagnation.
- Rewrite: completes in 575 seconds, returned 13 matches actual changed files. Twelve outputs require semantic
  review. They remain quality-pending, not negatives or accepted training rows.
- Article QA: completes in 49 seconds with “Dulce River” against “Dulce”. Source-aware equivalence remains pending
  independent adjudication. Zero token overlap with a rubric also defers (for example car/automobile).
- Repair replay previously dropped recorded token usage, changing context calibration and forced-compaction timing.
  Preserving usage restores identical observations; the genuine SMS classification failure produces one verified
  repair handoff. Queue-v5 prioritizes it before continuing the remaining work.
- Runtime-v7 and fresh output paths preserve old partials. The legacy extraction attempt was interrupted during
  migration and remains pending. Queue-v5 has 559 entries, 20 completed attempts and 539 pending at migration.
  Node/browser build and browser type checks pass, with 45 replay/oracle/collector regressions and 17 follow-up
  folder/handoff/oracle regressions passing. Earlier 132 focused regressions also passed.

### Follow-up: scheduling, shared state and payment scope

- A 142-message task with separate planning needs at least 284 requests for one judgment per item, plus root work.
  The old Luna cap of 256 could not finish that strategy. Moreover, concurrent siblings queued their planning
  requests first, consuming the budget before paired actions completed. Teacher turns now retain a concurrency
  slot across their plan/action pair. Requests count only after capacity admission; reaching the cap does not
  cancel already admitted work. A capped regression saves completed children and resumes to a correct result.
- Concurrent atomic checkpoint renames could finish out of order. Writes are serialized so an older snapshot
  cannot replace newer progress. This is a preventative fix; the audit did not establish a specific lost reply.
- The grants predicate decremented or reset a mutable captured budget, corrupting sibling judgments. Openings
  now explain that captured writes affect the caller/siblings, and that judgment calls should calculate with
  fresh locals unless instructed to update shared state. Mutable state remains available when the task needs it.
- Payment gold includes unrecognised card payments, direct debits and cash withdrawals. The earlier word “charge”
  did not make all those categories clear. Generator/reference prompts now state the scope explicitly, with
  `payment_scope_version: 2`; legacy BANKING77 mixed tasks are quarantined. Fourteen existing repair tasks migrate
  under new IDs/provenance, retaining their source groups and annotated gold. All fourteen references verify.
- Folder guidance asks agents to parse source tables and join judgments by ID, avoiding transcribed amount maps.
  It also recommends passing a large file handle and searching relevant portions rather than embedding the whole
  file in an NL instruction. Direct judgment and three ad hoc layers remain available.
- One Luna worker now uses the bounded supervisor, queue-v2 and fresh v2 jobs. Caps are sized to item counts
  (256–748 requests for this queue); deadlines range from 900–3,600 seconds with the same five-minute inactivity
  guard. One already admitted task is excluded, leaving 153 retries; three legacy source cases await reconstruction.
  Older partials/results remain preserved. Node/browser builds, browser type checks and 100 focused tests pass.

### Payment source review and criterion preservation

Payment0's three missed positives account for the full 1547 difference. One clear execution error is the child
adding exclusions/stronger evidence to the parent criterion: a question about a strange £1 transaction is still
about that transaction. The main tool prompt now asks delegating callers to preserve the criterion and context.
This guidance is absent at the third ad hoc layer and has not yet been deployed to the frozen v9 collectors.

Three BANKING77 annotations are pending review in `ts-host/src/teacher/source-review.ts`: cash shown but not
received; an app-recorded ATM withdrawal without an explicit dispute; and a generic debit-dispute question
annotated as an unrecognized direct debit. The registry preserves texts/labels and recognizes both current
text IDs and legacy text+label IDs. Main generation excludes them; admission, handoffs and preference building
hold tasks containing them. This defers ambiguity without relabeling from a teacher answer or counting it as
a verified model negative. Independent adjudication is still outstanding.

The 142-message payment case finished in 1060.5 seconds: 141 judgments match gold, with one missed 502 payment
giving 13386 against 13888. Benchmark tolerance accepts it; training admission excludes partial agreement
and pending-source tasks. The worker did finish useful work after the scheduling/budget changes. Both live
supervisors remain on runtime-v9; frozen-runtime migration is pending. Node/browser build, browser type checks
and 38 focused checks pass. No training has started.

### Runtime-v10 is active

Both workers now use frozen runtime-v10 (source d69af3a; 455 hashes verified), so criterion-preservation and
source-review quarantine apply during collection as well as main admission/export. Luna queue-v3 has 150
eligible entries, Bonsai queue-v7 has 558; three/one affected tasks were held, respectively. Source labels and
raw artifacts remain unchanged. Journals are reused; completed artifact paths remain accessible, pending
jobs use .v10 paths to avoid silently reusing old prompt history. Luna remains one worker and one request.

Both final v9 cases reached their existing hard deadlines: Luna CUAD extraction saved 106 replies with an
output CSV under verification; Bonsai handoff 555 saved 18 fresh replies. These are incomplete checkpoints,
not accepted training rows or verified model negatives. Their retries still need review. No independent
annotation adjudication or model training has started.
