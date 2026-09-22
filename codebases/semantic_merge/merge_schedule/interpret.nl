---
description: Semantically reconcile event times, locations and identity.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
Read each author request in the event's context. A reschedule may refer to the same event
even when the title changes. Distinguish alternative time proposals from two distinct
events. Treat time-zone or date ambiguity as unresolved unless the supplied data settles
it; do not invent a zone. Preserve stable event IDs, account for all update IDs, and
explain bookings that cannot both be honored under `policy`.
