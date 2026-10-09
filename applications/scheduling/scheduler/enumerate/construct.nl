---
description: Build complete feasible schedules by depth-first search over the tasks in dependency order.
args:
  domains: Domain[]
  order: string[]
  slot: number
  limit: number
returns: Offered
---
Build complete schedules, up to limit of them, one start per task, in eval. Minutes are counted from the day's origin.
Keep: placed, an object from task id to { start, end }, empty at first; options, an array, empty at first; truncated,
false at first.

Write a recursive function place(remaining) that places the task order[order.length - remaining] and then calls
place(remaining - 1). Its steps:
1. When remaining is 0, every task is placed: when options.length is below limit, push
   { id: "c" + (options.length + 1), placements: order.map(id => ({ id, start: placed[id].start, end: placed[id].end })) };
   otherwise set truncated to true. Return.
2. When truncated is true, return.
3. Let d be the domain of that task. ready is the largest end among placed[x] for the ids x in d.after, or 0 when
   d.after is empty.
4. For each span in d.spans, ascending: first = ceil(max(span.start, ready) / slot) * slot. The starts to try are
   first, first + slot, first + 2 * slot, and so on, while start + d.minutes is at most span.end; compute how many
   there are, count = floor((span.end - d.minutes - first) / slot) + 1, and use a counted for loop over that many.
5. For a start s with end e = s + d.minutes: skip it when some task already in placed has start below e and end above
   s. Otherwise set placed[d.task] = { start: s, end: e }, call place(remaining - 1), and delete placed[d.task].
   After the call, when truncated is true, return.

Call place(order.length). Return { options, truncated, detail } with detail "no feasible complete schedule" when
options is empty and "" otherwise.
