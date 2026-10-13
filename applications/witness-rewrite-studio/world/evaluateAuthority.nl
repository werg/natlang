---
description: Separate selected-world facts from belief and protected uncertainty.
args:
  intake: IntakeResult
  frames: FrameIndex
  entities: EntityIndex
  temporal: TemporalGraph
returns: AuthorityView
---
1. Read controlling canon and selected scene declarations before judging event claims. Identify
   authorityKind from the source's role, not how confidently an actor speaks.
2. For each selected event, convert narrator-established action into AuthorizedFacts with visible
   IDs, frame and exact witnesses. ObservationMeanings establish only what access actually permits.
3. Put remembered motives, unendorsed reports and speculative future claims in beliefsOnly unless
   a separate controlling source establishes them as facts. Retain speaker/frame provenance.
4. Use explicit director rulings to resolve competing takes. An unselected take can explain a draft
   history, but cannot supply a missing physical transfer in the selected story.
5. Create UncertaintyFacts for competing source-supported readings. Mark protected true only when
   intake.meaning.rules explicitly require those alternatives to remain unresolved.
6. Keep a protected motive uncertainty even if one interpretation would simplify later reconciliation.
   Changing an action does not automatically select a remembered explanation of that action.
7. When selected facts conflict with canon, expose the conflicting authority and witnesses as an
   uncertainty. Do not silently hide whichever assertion appears in the shorter file.
8. Return facts, beliefsOnly and uncertainty. Use entity and temporal results only for compatible
   identity/time joins; neither changes a statement's source authority by itself.

9. Interpret controlling canon records directly into AuthorizedFacts with their visible fact IDs,
   selected-world scope and exact quotations. Consume declared frames only to bind scope: canon
   definitions of letter contents, key capabilities and bells are not extra scene EventViews.
   Bind each canon witness from its original SourceRecord id/path, then check its exact quotation
   before joining it with selected events. Preserve stable canon even when no scene repeats it.
