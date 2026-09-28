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

Bonsai uses frozen `runtime-v5`, `queue-v4.jsonl` and the existing append-only journal. Fresh jobs use distinct paths
so old rows and partials are preserved. Default limits remain one case at a time, two model requests in flight,
128 requests and 600 seconds. Multi-item FOLIO/ANLI gets 1,200 seconds; all-file tasks get a size-based deadline up
to 3,600 seconds and up to 512 requests. All retain the inactivity guard and 20 turns per call. The user's three
ad hoc layers per existing file root remain unchanged. Luna stays stopped; no model training has started.
