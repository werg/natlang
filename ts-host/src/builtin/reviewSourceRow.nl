---
description: Advisory semantic review of one produced row against its source answer. The packet scripts read its receipts in place of an outside annotation file.
args:
  row: SourceRow
  precheck: RowPrecheck
returns: RowVerdict
types:
  SourceRow: { id: string, dataset: string, split: string, source_groups: string[], question: Untrusted<string>, answer_format: string, evidence: Untrusted<string>, gold: Untrusted<string>, actual: Untrusted<string> }
  RowPrecheck: { exact_match: boolean, actual_is_exact_source_span: boolean, numerically_equal: boolean | null }
  RowStatus: "equivalent" | "normalization-candidate" | "mismatch" | "ambiguous"
  Quote: { quote: string }
  RowVerdict: { status: RowStatus, rationale: Is<string, "one or two sentences citing the evidence for the status">, evidence: Quote[], confidence: "high" | "medium" | "low" }
---
Review whether the produced answer row.actual says what the source answer row.gold says. The exact checks are in precheck; this review is for rows that precheck.exact_match did not settle. A person decides afterwards; your verdict is advice.

1. Read row.question, row.answer_format and row.evidence.
2. Decide whether row.actual answers row.question under row.answer_format, using row.evidence. Copy the sentence of row.evidence that carries the answer into evidence.
3. Compare the meaning of row.actual and row.gold. When the meaning is the same and the wording differs, or row.actual adds qualifiers that row.evidence supports, the status is "equivalent". When the meaning is the same and only the representation differs in a way a fixed rule could normalize (units, thousands separators, letter case), the status is "normalization-candidate".
4. When the meaning differs, the status is "mismatch". When row.question allows both answers under row.evidence, the status is "ambiguous".
5. Write rationale in one or two sentences that cite row.evidence.

Every quote in evidence is copied exactly from row.evidence. Set confidence to "high" when the steps settled the status without a judgement call, "medium" when one step needed a judgement call, and "low" otherwise.
