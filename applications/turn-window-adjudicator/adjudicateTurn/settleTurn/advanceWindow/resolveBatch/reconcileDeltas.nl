---
description: Join same-object simultaneous proposals and derive departures from their combined damage.
args:
  bundle: EventBundle
  outcomes: PartOutcome[]
  beforeBoard: BoardSnapshot
  source: TurnSource
returns: BatchDelta
---
1. Group actual proposed changes by physical instance or player resource. For each proposal check its
   before value against the common beforeBoard; that exact integrity check precedes semantic merging.
2. Read the source's simultaneous accumulation rule. Add independent damage increments and growth
   increments separately; preserve integrity and controller unless a distinct actual effect changes
   them. Ink additions to one recipient accumulate without changing another player's resource.
3. If two nonadditive proposals assign incompatible controller, attachment, zone or new-instance
   properties, use only a supplied source precedence or explicit simultaneous rule. Record unresolved
   conflict where that source rule is absent; storage order cannot select a winning assignment.
4. Construct one merged ObjectChange per affected instance and one ResourceChange per affected player.
   Then inspect combined damage against source integrity for every affected garden creature. A creature
   meeting its destruction condition departs in this same batch, even when several parts supplied damage.
5. For each such departure, record its original instance/card IDs, controller immediately before the
   zone change, contributing part IDs, source witnesses and before object. Set its merged after zone
   to discard. Preserve the old instance's identity in history for subsequent last-known references.
6. Reconcile attachment consequences only when the source supplies them. A later return cannot be
   folded into this batch's destruction step; it is a separately queued effect with its own snapshot.
7. Return the merged BatchDelta, actual outcomes, departures, empty replacement-token list for its
   parent to fill, exact witnesses and unresolved conflicts. No board mutation occurs in this function.
