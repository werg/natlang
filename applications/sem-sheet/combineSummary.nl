---
description: Combine two partial column summaries into one concise summary for tree-mode semanticReduce.
args:
  goal: string
  left: string
  right: string
returns: string
---
Combine the two partial spreadsheet column summaries left and right into a single concise summary serving goal.
Keep every concrete fact, metric, or theme present in either side without repeating duplicates.
Return only the merged summary text.
