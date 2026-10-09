---
description: Work out where each task of the day can go, once fixed commitments, windows and bounds are taken into account.
args:
  view: DayView
  hard: Hard
returns: CheckedDomains
---
Compute the places each task can go. Minutes are counted from view.origin. Do the arithmetic in eval with plain numbers.

1. Tasks: view.tasks followed by hard.tasks, each as a copy with earliest, latest and after. For every entry of
   hard.limits, find its task and then: raise the task's earliest to limit.notBefore when that is larger; lower its
   latest to limit.endBy when that is smaller; add each id of limit.after that the task's after does not hold yet.
2. Busy time: collect view.fixed and hard.blocks as { start, end }. Sort them by start. Walk them in order and merge a
   span into the one before it when its start is at most that one's end, so the end becomes the larger end. The busy
   list is now ascending and disjoint.
3. Free time: for each window in view.windows, ascending, start with the piece { start: window.start, end: window.end }.
   Walk the busy list in order; for a busy span that overlaps the piece, the part of the piece before the busy start
   (when it has length) is a free span, and the piece continues from the busy end. When the walk ends, the rest of the
   piece (when it has length) is a free span. The free list is ascending and disjoint.
4. For each task, for each free span f in order: start = max(f.start, earliest) and end = min(f.end, latest). Round
   start up to a multiple of view.slot: start = ceil(start / slot) * slot. When end - start is at least the task's
   minutes, the span { start, end } is one of the task's spans.
5. Return one Domain per task, in the order of step 1: { task: id, minutes, after, spans }. A task with no span has
   spans [] and cannot be placed.
