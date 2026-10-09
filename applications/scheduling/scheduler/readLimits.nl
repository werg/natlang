---
description: Read the hard limits and fixed commitments a scheduling request states.
args:
  request: string
  view: DayView
  tasks: TaskView[]
returns: LimitReading
---
Read request, the user's words about their day, and list what it fixes: statements that every acceptable plan must
satisfy. tasks are all the tasks of the day, the existing ones and the ones the request adds. view.clock lists the
windows and commitments as clock times; view.origin is minute 0.

1. A limit narrows one task: "review not before 11:00" gives { task: "review", notBefore: <minute of 11:00>, reason };
   "draft done by 10:30" gives endBy; "review comes after the call" gives after: ["call"]. Set only the fields the
   request states. reason quotes the words of the request that the limit restates.
2. A commitment is time the user will spend on something else ("I'm at the dentist 11:00-11:30"). It becomes a block
   { id, start, end, reason }, with an id of one word (letters, digits, underscores, hyphens; a letter first) that is
   not the id of any entry in view.fixed.
3. A statement is firm when it uses words like "must", "not before", "I'm busy", "at 14:00 sharp". A wish
   ("preferably", "I'd like", "if possible") is a preference that another stage reads, so it is not listed here.
4. Every limit names a task in tasks. Times are whole minutes after view.origin: a clock time's minute is its distance
   in minutes from the origin's clock time.
5. When a statement could mean two different times or tasks, one question goes into questions and that statement
   waits for the answer. questions is [] when the request is clear.

Return { limits, blocks, questions }. A request with no firm statements gives { limits: [], blocks: [], questions: [] }.
