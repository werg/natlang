---
description: Bind exact event-relative windows without substituting a convenient timestamp.
args:
  ownerId: string
  trial: TrialData
  lineage: Lineage
  question: Question
returns: TimingView
---
1. Resolve this owner's selected anchor and interval clauses by lineage. Determine event type,
   occurrence selection, interval units, offsets and inclusion of each boundary.
2. For each unit interpret its commissioning and calibration events. A rejected commissioning retry
   is not a second accepted commissioning; a later calibration may have a different actual timestamp.
3. Select the event required by this owner's definition, preserving eventId, time and exact witnesses.
   Where multiple events remain eligible and the occurrence rule is absent, return null and unresolved.
4. Do not substitute firstTelemetryTime for an event anchor unless the selected clause specifically
   names telemetry. A convenient timestamp can produce the same baseline window by coincidence.
5. Interpret windowStart/windowEnd as offsets in the stated units. Preserve inclusive endpoints as
   includeStart/includeEnd; equal dates do not eliminate their meaning on boundary counterexamples.
6. Return anchorByUnit for the whole inventoried population, definitionIds and unresolved. Plans
   later use only included units, but excluded-unit anchors remain inspectable for changed probes.
7. Each anchor is a semantic choice backed by the source clause and event record. Exact subtraction
   and interval testing happen in the measurement service after this stage's choice is supplied.
8. A candidate repair can retain all raw times while changing which event owns the origin. Audit
   must inspect this view rather than infer correct anchoring from a restored headline percentage.
