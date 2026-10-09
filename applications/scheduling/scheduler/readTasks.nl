---
description: Read the new tasks a scheduling request introduces.
args:
  request: string
  view: DayView
returns: TaskReading
---
Read request, the user's words about their day, and list the tasks it introduces that view.tasks does not already hold.

1. A task is a piece of work the request asks to fit into the day ("add a 45 minute workout", "I also need to call
   Sam for 20 minutes after the review"). A request that only talks about tasks already in view.tasks introduces none:
   return tasks [].
2. Give each new task these fields:
   - id: one word from the task's name, starting with a letter, with letters, digits, underscores and hyphens only.
     It differs from every id in view.tasks and from the other new ids.
   - minutes: the duration the request states, a whole number.
   - earliest and latest: minutes after view.origin. The request's own bounds ("after lunch", "by 15:00") when it gives
     them; otherwise earliest is the start of the first window in view.windows and latest is the end of the last.
   - after: the ids of the tasks the request says this one follows. Empty when it names none.
3. Convert a clock time to minutes with view.origin and view.clock: view.origin is minute 0 at its clock time, so a
   clock time that is 90 minutes later than the origin's is minute 90. view.windows are already in minutes.
4. A task is included once the request states its duration. When it does not, one question goes into questions that
   asks for it ("How long should the workout take?"). The same holds for a bound with two possible readings.
   questions is [] when the request is clear.

Return { tasks, questions }.
