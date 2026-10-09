---
description: Explain why no schedule exists and what the user could relax.
args:
  request: string
  view: DayView
  hard: Hard
  domains: Domain[]
returns: Diagnosis
---
No complete schedule satisfies the day plus what request adds (hard). Find the reason. view.clock lists the windows and
commitments as clock times; view.origin is minute 0; hard and domains are in minutes after the origin.

1. A task whose domain has no span cannot be placed at all: its conflict names the task, its minutes, its bounds
   (earliest, latest, after limits) and the commitments or window edges that leave no room, as clock times.
2. When every task has a span, look for a shortage: the minutes of the tasks that compete for the same spans against the
   length of those spans; and dependency chains (a task that must follow another) whose total minutes exceed the room
   between the chain's first earliest and last latest. Name the tightest one.
3. conflict is one or two sentences that name the tasks or commitments that cannot all hold, with their clock times.
4. relaxations lists, most promising first, up to three changes the user could make: move a commitment, widen a
   window or a bound, shorten a task, or drop a dependency. Each is a sentence naming the task or commitment and the
   change.

Return { conflict, relaxations }.
