---
description: Reconcile claims only against observed sources and their provenance.
args: { request:CommissionRequest; sources:ArchiveSource[]; observations:SearchObservation[]; leads:FollowupLead[]; checkpoint:ResearchCheckpoint }
returns: SupportedClaim[]
---
Build claims only from source identifiers in hit observations. Separate recorded wording from interpretation, and link contradictory sources explicitly. A no-hit supports only a coverage statement for the precise search. Missing-source leads cannot prove an event. Carry unchanged claims only if all cited source revisions still match.
