---
description: Write the answer as prose over the claims.
args:
  question: string
  claims: Claim[]
  gaps: string[]
returns: string
---
Write the answer to question.

1. Write two to four sentences that answer the question using only the claims. When claims is empty, write one sentence
   saying that the sources do not answer the question.
2. When gaps is not empty, end with one sentence about the strongest gap.
