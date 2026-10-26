---
description: Build a time-indexed production resource ledger and independently check consent paths and resource collisions.
args: { production: ImmersiveProduction; dramaturgy: DramaturgyMap; incident: ShowIncident; showEvidence: EvidenceItem[]; safetyAndConsentRules: EvidenceItem[] }
returns: ResourceTimeline
---
1. Bind windows = buildResourceWindows(production, dramaturgy.scenes, showEvidence). Preserve separate entries for the person, their qualification, their actual availability, and any required equipment or accessible route.
2. Bind collisions = findOverlappingReservations(windows), then consent = verifyConsentPath(incident, dramaturgy.scenes, showEvidence, safetyAndConsentRules). A free resource does not establish consent, and consent does not establish capacity.
3. Return ResourceTimeline { windows, conflicts: collisions.conflicts plus consent.blockers, consentRoutes: consent.routes, unavailable: collisions.unavailable, unresolved: collisions.unverified plus consent.blockers }. Retain unavailable resources even if another scene might use them. The caller must choose only variants that avoid their demands. Missing current status remains unverified.
