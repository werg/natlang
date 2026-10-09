---
description: Explain a chosen schedule to the user, saying what it satisfies, what it gives up, and what it assumed.
args:
  request: string
  schedule: string[]
  preferences: Preference[]
  assessments: Assessment[]
  considered: number
  truncated: boolean
returns: string
---
Write the explanation the user reads next to their plan. request is what they asked, schedule the plan (one line per
task with clock times), preferences their wishes and assessments how the plan meets each (an Assessment per wish:
fit is met, partly or missed).

1. First sentence: the plan in one line, naming the first and last task and their clock times.
2. One clause per wish with fit met, saying what the plan does for it, in the user's words.
3. One clause per wish with fit partly or missed, saying what the plan gives up and why a better arrangement did not
   exist among the plans compared (the hard limits and commitments held the tasks where they are).
4. When truncated is true, say that more plans exist than the considered ones that were compared.
5. Use clock times, plain words and at most five sentences. Wishes are mentioned by their text.

Return the text.
