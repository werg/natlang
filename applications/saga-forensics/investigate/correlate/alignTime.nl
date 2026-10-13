---
description: Interpret clock clauses and normalize only events covered by those clauses.
args:
  index: EvidenceIndex
  asOfGlobalTime: number
returns: Timeline
---
1. Collect clock atoms and any contract or boundary clause that limits their applicability. Call interpretClock
   once for each explicitly named clock, passing its supporting atoms and the named clock.
2. Enumerate every explicitly timed event mentioned in attempts, receipts, handover, cancellations and queries.
   An event and a report of that event may have different timestamps; keep the receipt-report time distinct.
3. For each event, retain its visible eventId, localTime, clock and kind. Use null for an unstated local time.
4. Find the returned ClockRelation whose scope covers that event. If exactly one supported offset applies,
   compute globalTime = localTime + globalOffset in eval. Preserve the clock and event witnesses together.
5. If an event is already explicitly in global time, use its stated time and the identity global-clock relation.
   If two applicable relations disagree or none supplies an offset, leave globalTime null and explain why.
6. A restart or epoch change modifies the clock only when a source clause says it does. Do not import that
   assumption from the separate request-key namespace rules.
7. Interpret query coverageThrough and processing boundaries as event kinds so later query selection can
   compare them on the same scale. Coverage through time T includes T unless its clause says otherwise.
8. Record post-snapshot events as unresolved for this investigation; do not silently treat them as observed.
9. Return all clock relations, event times and unresolved explanations. Arithmetic is exact after interpretation.
