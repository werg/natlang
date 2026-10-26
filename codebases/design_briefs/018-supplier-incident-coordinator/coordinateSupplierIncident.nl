---
args: { request: SupplierIncidentRequest, trace: SupplierTraceFile, prior?: SupplierIncidentCheckpoint|null }
returns: SupplierIncidentResult
uses: [incidentDesk/frameIncident, incidentDesk/selectIncidentAuthority, incidentDesk/clusterIncidents, incidentDesk/resolveLotAliases, incidentDesk/traceIncidentExposure, incidentDesk/planIncidentContainment, incidentDesk/escalateMissedResponse, incidentDesk/testSupplierCorrection, incidentDesk/reopenCausalRecurrence, incidentDesk/reviewIncidentClosure, incidentDesk/amendCorrectedNotice]
---
Bind saved = prior when supplied, otherwise null.
1. Bind goal = frameIncident(request, trace.incidents). Bind policy = selectIncidentAuthority(request, trace.policyCandidates). Bind clusters = clusterIncidents(request, trace.incidents, trace.evidencePages). Preserve incident and report versions.
2. Bind aliases = resolveLotAliases(request, trace.lotAliasEvidence) before tracing; keep original lot labels and authenticated renumber mappings.
3. Bind exposure = traceIncidentExposure(request, clusters, trace.evidencePages, aliases, trace.splitStockTrace). Split-stock evidence constrains the exposed quantity; unknown lineage stays precautionary.
4. Bind escalation = escalateMissedResponse(request, trace.responseDeadlineEvidence). A missed supplier deadline escalates by policy but does not make its proposed explanation accepted.
5. Bind containment = planIncidentContainment(request, exposure, policy, trace.evidencePages). Do not construct a release action before the exposure and authority are supported.
6. Bind correction = testSupplierCorrection(request, clusters, exposure, containment, trace.evidencePages + trace.recurrenceEvidence). The supplier's explanation must address a supported mechanism and distinguish recurrence prevention from immediate correction.
7. Bind recurrence = reopenCausalRecurrence(request, trace.recurrenceEvidence). If materially similar, preserve old closure and require a new test before closing again.
8. Initialize closure: StageResultIncidentClosure|null = null. When containment and corrective verification evidence are complete, bind closure = reviewIncidentClosure(request, containment, correction, recurrence, trace.evidencePages). Otherwise preserve null closure and keep monitoring and residual exposure open in the correction and exposure findings.
9. Bind notice = amendCorrectedNotice(request, trace.noticeCorrectionEvidence). Preserve original recipients and delivery receipts when proposing an amendment.
10. Bind checkpoint = reconcileIncidentCheckpoint(saved, request, trace, clusters, exposure, containment, correction, closure, notice). Preserve original incident IDs and split-stock exposure lineage; keep hold, supplier notice, correction proof, and closure as separate receipt families. Reopen only lots touched by new exposure or causal recurrence evidence. A proposed notice or corrective action is not an executed effect.
11. Return { caseId: request.caseId, goal, policy, clusters, exposure, containment, correction, closure, guardrails: [aliases, escalation, recurrence, notice], checkpoint, commitAllowed: false }.
