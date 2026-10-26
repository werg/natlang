---
description: Commission and resume a bounded archival inquiry from an evidence and search checkpoint.
args: { sourceRecord: ArchiveCommission; sourceEvidence: EvidenceItem[]; request: CommissionRequest; sources: ArchiveSource[]; observations: SearchObservation[]; checkpoint: ResearchCheckpoint }
returns: CommissionDossier
---
Bind commission = `bindCommissionScope(request, sourceRecord, sourceEvidence)`. Its research question, requester, researcher, collection IDs and effort budget come directly from the record top level; do not read them from record.fields. If commission.blockers is nonempty, preserve those exact source-reference gaps in the dossier coverage limits and do not claim a completed search.

Bind bounded = `boundQuestions(commission.request, checkpoint)` before any search. Set questions = bounded.questions. Each question must have a plausible evidence path, an inferential ceiling, and a finite stopping rule. Reuse unchanged checkpoint questions; changed scope reopens dependent findings.

Bind routed = `routeArchiveSearch(commission.request, sources, questions, checkpoint.manifest, observations)` to retain actual observation records and question-specific collection coverage. Set observationsForDossier = routed.observations; pass routed.coverage unchanged to dossier assembly. Never treat routed coverage plans as source hits or completed searches.

Bind leads = `traceSourceLeads(commission.request, sources, observationsForDossier, checkpoint)`. A catalog reference to absent material creates a missing-source lead with unknown contents. Compare citation ancestry so copied citations cannot masquerade as independent witnesses. Pursue only leads within remaining effort; when the next valuable lead exceeds budget, preserve it and set extension-needed.

Bind claims = `weighArchiveClaims(commission.request, sources, observationsForDossier, leads, checkpoint)` to reconcile actual source observations, chronology, attribution, contradiction and inference limits. Return the complete `composeResearchDossier(commission.request, sources, observationsForDossier, routed.coverage, leads, claims, checkpoint, commission.blockers)` result so collection coverage and scope-binding evidence gaps reach the final dossier and incomplete status. A negative claim applies only to recorded query coverage; never imply a search occurred without its supplied observation.
