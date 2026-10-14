---
description: Interpret the exact scope of one rehearsal relation over actual performed events.
args:
  constraint: TimingConstraint
  events: PerformedEvent[]
returns: ConstraintScope
---
1. Create a scoped NL ConstraintScope literal and directly await it on this actual constraint and
   actual performed events. Its first step joins the source section/event identities to performances.
2. Its second step decides what the witness actually constrains: an ending sonority describes sounds
   active together, whereas attack synchronization describes onset times. It preserves an unordered
   observed rank pair unless the source explicitly identifies the individual voice owners.
3. Its third step returns applicability true/false/null, actual original event IDs involved, a reason
   and exact evidence. A copied observation adds no independent testimony; missing event scope stays null.
4. Validate the returned IDs against these events and witness support against the actual constraint
   and event evidence. Consume the actual scope result before the parent performs exact tick/rank checks.
5. Return that scoped result unchanged. Numerical agreement alone cannot broaden a witness's scope
   or make an unspecified voice assignment into an established observation.
