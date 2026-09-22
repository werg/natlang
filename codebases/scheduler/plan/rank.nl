---
args:
  request: string
  snapshot: ScheduleSnapshot
  alternatives: Alternatives
returns: Candidate
---
Choose one offered complete candidate by ID and slots. Interpret the user's
soft preferences in the request and explain any tradeoff. Hard constraints
have already been checked exactly. Do not invent a new slot or alter exact
times. If alternatives.truncated is true, mention that the ranking saw only
a bounded subset. Return exactly one offered candidate.
