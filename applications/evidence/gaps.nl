---
description: Name the contradictions between claims and the parts of the question no claim addresses.
args:
  question: string
  claims: Claim[]
  passages: Passage[]
returns: string[]
---
Write the gaps in what the claims establish about question.

1. Compare the claims that speak about the same thing. For each pair that disagrees, write one line naming both span ids.
2. List each part of the question that no claim addresses, one line each.
3. Return the lines as a list of strings; the list is empty when the claims agree and cover the question.
