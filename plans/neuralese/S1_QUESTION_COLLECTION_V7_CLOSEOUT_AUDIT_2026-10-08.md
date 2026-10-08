# S1 Question Collection V7 Closeout Audit

**Review date:** 2026-10-08
**Purpose:** source and question review of the V7 closeout; no change to frozen targets, no corpus admission, and no global substring rule.

## Pinned evidence

- Source cases: `runs/s1-question-collection-candidates-20261008-v7/source.cases.jsonl`, SHA-256 `091044d6069ee1480d645ea308c2625933b2c76f8a8530a51933a20790ca020f`.
- Closeout: `runs/s1-question-collection-candidates-20261008-v7/generation-review-v1/luna/campaign-v1/closeout-review-v1.json`, SHA-256 `0deb0d43de218916a08fb1aa4d3b255914e37b4ec3b054b80bd559346b0587d1`.
- Per-field question/output hashes and reviewed source-bound equivalence candidates are in [the review receipt JSON](S1_QUESTION_COLLECTION_V7_SEMANTIC_REVIEW_2026-10-08.json).

The closeout reports 15 completed cases, one provider transport failure, 31 exact fields of 51, ten longer non-exact answers that are exact source spans and contain the gold string, plus other mismatches. A completed host run is not itself a semantic pass. The transport failure at source index 15 has no answer and is excluded from semantic judgment.

## Source-supported answer variants

The ten longer answers are from source indices 0–2, held across three distinct source groups in the `test` split. Their `answer_format` says to return exact source wording without explanation or added qualifiers. The answer helper does not define how much of a grammatical source phrase belongs in the answer, while the source targets often clip that phrase to a bare value. Inspection of each question and evidence found the returned phrase directly answers the question and introduces no unsupported fact:

| Source index / field | Gold → actual | Review |
|---|---|---|
| 0 / item-02 | `a copper statue of Christ` → `a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes"` | Describes the requested object using details in the source. Longer than the clipped gold answer, but supported and responsive. |
| 0 / item-03 | `Rome` → `in Rome` | Exact location phrase from the passage; the preposition makes a natural answer to “Where”. |
| 0 / item-04 | `1925` → `won the Rose Bowl in 1925` | Exact event phrase with the requested year. |
| 1 / item-01 | `$250,000` → `an initial $250,000` | Exact amount phrase; “initial” mirrors the question. |
| 1 / item-02 | `expanded to work with other charities` → `The foundation has since expanded to work with other charities in the city` | Exact source sentence answering how the foundation changed. |
| 1 / item-04 | `$250,000.` → `an initial $250,000` | Same source-supported amount; the mismatch is answer phrasing/punctuation, not amount. |
| 2 / item-01 | `Spanish word montaña (mountain)` → `derived from the Spanish word montaña (mountain)` | Source phrasing directly answers the origin question. |
| 2 / item-02 | `4th` → `4th in size` | Source phrase answers the requested size rank. |
| 2 / item-03 | `44th` → `44th in population` | Source phrase answers the requested population rank. |
| 2 / item-04 | `77` → `77 named ranges` | Exact source phrase supplies the count and the noun requested. |

These are per-field semantic-equivalence candidates, not exact string matches. The receipt binds each answer to its source case, question, result, and trace hashes. A future semantic lane may use these reviewed receipts; exact-target data remains unchanged. The receipt schema requires exact source/question/output hashes and does not authorize accepting arbitrary answers merely because they contain a gold substring.

Three other source-supported mismatches merit the same separate semantic review: source index 3 / item-03 omits only quote marks around `in whole or in part`; source index 3 / item-04 gives the full 2001 *Prosecutor v. Radislav Krstic* Trial Chamber I judgment citation that the supplied passage identifies as the genocide case; source index 13 / item-01 says `Belfast journalist Deric Henderson's account`, directly answering whose account the drama used. These outputs are included as candidates in the receipt JSON, with source and question hashes.

Source index 15 has a separate successful infra-recovery retry after the original campaign hit a provider transport failure. The retry answers `Middleweight division` for the gold `the Middleweight division`; the source passage says Jay Silva is competing in “the Middleweight division.” The only difference is the article. This is an exact match under ordinary SQuAD/Hotpot answer normalization (lowercase, strip punctuation and articles, collapse whitespace), not a substring heuristic. The receipt keeps the retry provenance separate from the failed original attempt. `ts-host/src/skills/graded.ts` already contains SQuAD-style token normalization; a shared corpus grader should reuse the same normalization contract rather than enforcing the adapter's hidden article boundary.

## Cases held outside the equivalence receipt

- Multiple-choice mismatches at source indices 5, 6, and 7 select labels other than the gold choices. Source-span checks do not establish those labels as equivalent; keep them strict.
- Source index 8 / item-03 is ambiguous with the available data: the department table has 15 entries and the management table lists four distinct department IDs, so 11 departments have no management row. But no `head` table is supplied, and the question says “heads who are not mentioned.” The model returned `not determinable`. Hold both gold and output for clarification of the intended relation.
- Source index 11 / item-04 is a likely source/label ambiguity: the table contains 14 distinct allergy names and 3 distinct allergy categories. The question asks “How many distinct allergies,” while gold `3` counts categories. The actual `14` is supported by the literal wording, but is not equivalent to `3`; hold for a specific semantic interpretation.
- Source index 9 / item-04 returns `3977.7500000000005` for a gold `3977.75`. Its child eval computed `(3900.1 + 4090.5 + 4198.8 + 3721.6) / 4` and returned `String(avg)`. The exact decimal mean is `3977.75`; JavaScript binary64 accumulation introduced a one-ULP display artifact at this magnitude. Keep it held for canonical exact repair. Do not add a generic `1e-12` numeric acceptance rule. Future numeric tasks should make exact decimal arithmetic or precision-aware formatting an explicit task requirement where the source precision supports it.

For future generated question items, a concise-answer instruction should ask for the smallest complete source-supported answer, including the needed noun, unit, relation, or disambiguating context. Avoid combining a generic request for “exact source wording” with targets clipped to unexplained fragments. When a task expects exact computation from decimal source values, spell out the precision policy or direct the solver to exact decimal arithmetic; never round through an unspecified global tolerance.
