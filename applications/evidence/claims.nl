---
description: Write the claims that the passages support, each with the passage id and a quote copied from the passage.
args:
  question: string
  passages: Passage[]
  problem?: string
returns: ClaimDraft[]
---
Write the claims that passages support about question.

1. For each passage, find the sentences that bear on the question.
2. For each such sentence write a claim: one sentence in your own words (text), the id of the passage (span_id), and a
   short stretch of words copied from the passage (quote).
3. Keep a claim when its quote supports it. The quote shows where the claim came from.
4. problem, when given, says why an earlier list was refused. Write the list again so that each span_id is the id of
   one of passages and each quote is copied from the passage with that id.
