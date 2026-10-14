---
description: Resolve source readings per written location while preserving supported alternatives.
args:
  archive: Archive
  families: FamilyIndex
returns: ReadingIndex
---
1. Form the finite original event ID universe from actual partbook readings. A missing decisive
   original event or unresolved source inventory stays visible; do not create a convenient replacement.
2. Bind immutable eventOrder to those original event IDs in the chosen traversal order.
   Initialize ArbitrationFrontier with a separate pending copy of eventOrder, empty bundles, actual archive/families and
   accumulated source issues. Source order can choose traversal order, never semantic preference.
3. Invoke advanceReading.iterateOn with measure pendingEventIds.length. Consume its actual bundles,
   selected/rejected source readings and target decisions rather than reconstructing choices here.
4. Check exactly one bundle exists per original event ID. Every option must retain that event's
   original identity and source evidence, including applied correction and target-assignment support.
5. Return ReadingIndex with bundleOrder equal to eventOrder and bundles equal to the actual reducer result. Independent ensemble observations can later reject a globally impossible
   timeline; this stage preserves every locally supported option until that actual check occurs.
