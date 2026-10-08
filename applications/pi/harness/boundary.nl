---
description: Turn boundary selection (pi-durable inbox.ts applyBoundary and isStale, spec §6, appendix 2 R2). Which queued items a boundary places; the place operation then writes them.
args:
  items: InboxItem[]
  steeringMode: QueueMode
  followUpMode: QueueMode
  at: '"postTools" | "final"'
  activeStart: number | null
returns: BoundarySelection
---
Decide what a turn boundary places from the inbox items (in inbox order, the order given). activeStart is where the
active context starts (the newest head marker's head, or null when unknown). Nothing is written here: the caller
commits { op: "place", selection } with the result.

1. reset = some item has mode "write" and entry.head === "self".
2. final = at is "final", or reset (a queued reset makes a postTools boundary final).
3. writes: every item with mode "write", in inbox order. Judge each one's staleness in order, keeping start =
   activeStart up to date:
   - stale when its entry.head is a number, start is not null, and entry.head < start;
   - a stale write is listed with stale: true and changes nothing else;
   - a write that is not stale and has a head moves start: entry.head "self" means the new entry, which is newer than
     every entry that exists, so every later write with a numeric head is stale; a numeric head sets start to it.
   writes = [{ id, stale }] for every write item.
4. users: the steer items: all of them when steeringMode is "all", otherwise only the first one (the lowest ID). When
   final, also the followUp items: all when followUpMode is "all", otherwise only the first. List the chosen IDs in
   inbox order. Queued user items are never stale.

Return { writes, users, reset, final }.
