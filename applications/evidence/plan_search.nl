---
description: Turn a question into search phrases for the evidence index.
args:
  question: string
returns: SearchPhrases
---
Write search phrases for question.

1. List the key terms in the question: names, quantities and topics.
2. For each key term, add a likely synonym or alternative spelling.
3. Write two or three phrases, each combining two to four of those terms.
4. Return the phrases as a list of strings.
