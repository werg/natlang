---
description: Answer a question from the checked samples of the cells that ran.
args:
  question: string
  evidence: CellEvidence[]
  note: Untrusted<string>
  limits: string[]
  problem?: string
returns: string
---
Answer question from evidence. Each item of evidence is one cell: its id, its revision, its status, its sample (the cell's result, cut to a bounded length), and facts the host computed about the result: has_null, empty and truncated.

1. Read evidence. A sample is the result of the cell; has_null says the result contains NULL values, empty says the result has no rows or values, and truncated says the sample shows only the start of the result.
2. When limits is not empty, begin the answer by stating each limit in plain words, for example "cell totals failed, so this answer is incomplete".
3. Answer question from the samples. Write each cell you use as id@revision, for example totals@0.
4. Write a NULL as "NULL", an empty result as "no rows", and a field that a row does not have as "absent".
5. When note is not empty, use it as background and say that it comes from the note.
6. When problem is given, it states what was wrong with the previous answer; write an answer that fixes it.
