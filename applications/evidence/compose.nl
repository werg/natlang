---
args:
  question: string
  passages: Passage[]
  truncated: boolean
returns: Draft
---
Answer only what the passages establish. For every factual claim, include its
exact span ID, revision, and a short literal quote from that passage. Mention
contradictions and missing evidence in gaps. If retrieval was truncated,
include that limitation. A quote proves provenance, not semantic entailment;
do not overstate what it supports. Write exactly answer, claims, and gaps.
