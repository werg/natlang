---
description: Repair a near-miss schedule. Keep the placements that hold and move the ones the verifier names.
args:
  placements: Placement[]
  violations: Violation[]
  domains: Domain[]
  order: string[]
  slot: number
returns: CheckedPlacements
---
placements is a schedule that the verifier rejected, and violations says what it broke (each violation names a task and
the move that would fix it). Return a schedule with exactly one placement per task, in the order of `order`. Do the
work in eval with plain numbers. Minutes are counted from the day's origin.

Walk the tasks in `order`, keeping kept, an array of the placements settled so far, empty at first. For a task t with
domain d (the entry of domains whose task is t):
1. Take its current placement p from placements (the one with id t), if there is one.
2. p stays when all of these hold: no violation of violations names t; p.end - p.start equals d.minutes; p lies inside
   one span of d.spans (span.start <= p.start and p.end <= span.end); p.start is at least the end of every
   placement in kept whose id is in d.after; p overlaps no placement in kept (two placements overlap when one starts
   before the other ends and ends after the other starts). Push p on kept.
3. Otherwise t moves. ready is the largest end among the placements in kept whose id is in d.after, or 0. For each
   span of d.spans, ascending, try the starts ceil(max(span.start, ready) / slot) * slot, then each further slot, while
   start + d.minutes is at most span.end. The first start whose interval [start, start + d.minutes) overlaps no
   placement in kept is the new start. Push { id: t, start, end: start + d.minutes } on kept.
4. When no start exists, push the old placement p unchanged, or when t had none, { id: t, start: d.spans[0].start,
   end: d.spans[0].start + d.minutes } so the verifier reports what is still wrong.

Return kept.
