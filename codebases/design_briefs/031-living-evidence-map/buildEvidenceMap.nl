---
description: Maintain a versioned evidence map that links eligible studies, report lineage, outcome measures, claims, and review-changing events.
args: { request: EvidenceMapRequest, corpus: ResearchEvidencePack }
returns: EvidenceMapResult
uses: [evidenceMap/frameReview, evidenceMap/freezeEligibility, evidenceMap/linkStudyLineage, evidenceMap/screenAndExtract, evidenceMap/alignOutcomeMeasures, evidenceMap/buildClaimGraph, evidenceMap/reconcileRetraction, evidenceMap/rescreenChangedEligibility]
---
Bind saved = corpus.checkpoint when present, otherwise null. An absent checkpoint means this is the first review.

1. Bind review = frameReview(request, corpus.reviews) and eligibility = freezeEligibility(request, corpus.eligibilityPolicies). Freeze question, criteria version, corpus, and cutoff before screening.
2. Bind lineage = linkStudyLineage(corpus.studyRecords, corpus.versionLinks) before counting studies. A preprint and its journal report are linked versions of one study when provenance supports that relationship.
3. Bind observations = screenAndExtract(eligibility, lineage, corpus.studyRecords). Screen all records against the frozen criteria and retain design, population, outcome, estimate, unit, follow-up and limitations. Abstract-only records remain limited to statements actually present in the abstract.
4. Bind rescreenedObservations = rescreenChangedEligibility(eligibility, corpus.eligibilityChanges, lineage, observations). If criteria changed mid-review, use the new versioned screening ledger for any later claim map; preserve old screening history.
5. Bind measures = alignOutcomeMeasures(rescreenedObservations, corpus.measureConversions). Preserve original outcomes/units. Compare different units only when an approved exact conversion artifact is supplied.
6. Bind claims = buildClaimGraph(request, rescreenedObservations, measures, lineage). Link each proposition to supporting and opposing studies and their source spans; keep applicability limits and confidence rationale explicit.
7. Bind retractionUpdate = reconcileRetraction(claims, corpus.retractions) before returning the current map. Retracted reports invalidate dependent claims and create review-owner notices; unrelated claims remain intact.
8. Bind rescreenUpdate = summarizeEligibilityImpact(observations, rescreenedObservations, claims, corpus.eligibilityChanges). Retain which study observations changed and which existing claims require review; an absent eligibility change yields an explicit empty update.
9. Bind checkpoint = reconcileEvidenceMap(saved, request, corpus, review, eligibility, lineage, observations, rescreenedObservations, measures, claims, retractionUpdate, rescreenUpdate). Preserve the frozen question, cutoff and prior criteria editions. New retractions invalidate only dependent claims; changed eligibility criteria rescreen records under a new version and retain old screening decisions.
10. Return { review, eligibility, lineage, observations, rescreenedObservations, measures, claims, retractionUpdate, rescreenUpdate, unresolved: retractionUpdate.ownerQuestions plus rescreenUpdate.ownerQuestions, checkpoint, commitAllowed: false } as EvidenceMapResult. Preserve limitations; this map does not substitute for source retrieval or scientific review.
