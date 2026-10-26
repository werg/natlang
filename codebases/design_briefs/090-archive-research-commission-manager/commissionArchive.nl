---
description: Commission and resume a bounded archival inquiry from an evidence and search checkpoint.
args: { sourceRecord: ArchiveCommission; sourceEvidence: EvidenceItem[]; request: CommissionRequest; sources: ArchiveSource[]; observations: SearchObservation[]; checkpoint: ResearchCheckpoint }
returns: CommissionDossier
---
Bind commission = `bindCommissionScope(request, sourceRecord, sourceEvidence)`. Its research question, requester, researcher, collection IDs and effort budget come directly from the record top level; do not read them from record.fields. If commission.blockers is nonempty, preserve those exact source-reference gaps in the dossier coverage limits and do not claim a completed search.

Bind bounded = `boundQuestions(commission.request, checkpoint)` before any search. Set questions = bounded.questions. Each question must have a plausible evidence path, an inferential ceiling, and a finite stopping rule. Reuse unchanged checkpoint questions; changed scope reopens dependent findings.

Bind routed = `routeArchiveSearch(commission.request, sources, questions, checkpoint.manifest)` to retain coverage variants and distinguish inaccessible, closed, unsearched and exhausted collections. Combine routed results with supplied observations as observationsForDossier without duplicating prior entries.

Bind leads = `traceSourceLeads(commission.request, sources, observationsForDossier, checkpoint)`. A catalog reference to absent material creates a missing-source lead with unknown contents. Compare citation ancestry so copied citations cannot masquerade as independent witnesses. Pursue only leads within remaining effort; when the next valuable lead exceeds budget, preserve it and set extension-needed.

Bind claims = `weighArchiveClaims(commission.request, sources, observationsForDossier, leads, checkpoint)` to reconcile actual source observations, chronology, attribution, contradiction and inference limits. Return `composeResearchDossier(commission.request, sources, observationsForDossier, leads, claims, checkpoint)` to produce reproducible search history, access and reuse limits, and resume state. A negative claim applies only to recorded query coverage; never imply a search occurred without its supplied observation.
