---
description: Read the soft preferences a scheduling request states.
args:
  request: string
  tasks: TaskView[]
returns: PreferenceReading
---
Read request, the user's words about their day, and list the wishes it states: things the user would like, which a
good plan satisfies when it can. tasks are all the tasks of the day.

1. A wish is introduced by words like "preferably", "I'd rather", "if possible", "ideally", "I like", "keep ... free",
   or it is a task's own preference text (tasks[i].preference). Each task preference is a wish.
2. Give each wish these fields:
   - id: "p1", "p2", ... in the order the request states them.
   - text: the wish in one sentence, in the user's terms ("draft in the morning").
   - tasks: the ids of the tasks it concerns. Empty when it concerns the whole plan ("finish early", "leave the
     afternoon free").
   - weight: 3 when the request stresses it ("really", "important", "please make sure"), 1 when it is mild
     ("maybe", "a bit"), otherwise 2.
3. Firm statements ("must", "not before", "I'm busy") are limits that another stage reads, so they are not listed here.
4. When a wish refers to something the day does not hold, one question goes into questions that asks which task or
   time is meant, and that wish waits for the answer. questions is [] when the request is clear.

Return { preferences, questions }. A request with no wishes gives { preferences: [], questions: [] }.
