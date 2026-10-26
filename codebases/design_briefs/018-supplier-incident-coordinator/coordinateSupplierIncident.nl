---
args: { request: SupplierIncidentRequest, trace: SupplierTraceFile }
returns: SupplierIncidentResult
uses: [incidentDesk/frameIncident, incidentDesk/selectIncidentAuthority, incidentDesk/clusterIncidents, incidentDesk/resolveLotAliases, incidentDesk/traceIncidentExposure, incidentDesk/planIncidentContainment, incidentDesk/escalateMissedResponse, incidentDesk/testSupplierCorrection, incidentDesk/reopenCausalRecurrence, incidentDesk/reviewIncidentClosure, incidentDesk/amendCorrectedNotice]
---
1. Bind goal = frameIncident(request, trace.incidents). Bind policy = selectIncidentAuthority(request, trace.policyCandidates). Bind clusters = clusterIncidents(request, trace.incidents, trace.evidencePages). Preserve incident and report versions.
2. Bind aliases = resolveLotAliases(request, trace.lotAliasEvidence) before tracing; keep original lot labels and authenticated renumber mappings.
3. Bind exposure = traceIncidentExposure(request, clusters, trace.evidencePages, aliases, trace.splitStockTrace). Split-stock evidence constrains the exposed quantity; unknown lineage stays precautionary.
4. Bind escalation = escalateMissedResponse(request, trace.responseDeadlineEvidence). A missed supplier deadline escalates by policy but does not make its proposed explanation accepted.
5. Bind containment = planIncidentContainment(request, exposure, policy, trace.evidencePages). Do not construct a release action before the exposure and authority are supported.
6. Bind correction = testSupplierCorrection(request, clusters, exposure, containment, trace.evidencePages + trace.recurrenceEvidence). The supplier's explanation must address a supported mechanism and distinguish recurrence prevention from immediate correction.
7. Bind recurrence = reopenCausalRecurrence(request, trace.recurrenceEvidence). If materially similar, preserve old closure and require a new test before closing again.
8. Bind closure = reviewIncidentClosure(request, containment, correction, recurrence, trace.evidencePages) only when containment and corrective verification evidence are complete; otherwise keep monitoring and residual exposure open.
9. Bind notice = amendCorrectedNotice(request, trace.noticeCorrectionEvidence). Preserve original recipients and delivery receipts when proposing an amendment.
10. Return { caseId: request.caseId, goal, policy, clusters, exposure, containment, correction, closure, guardrails: [aliases, escalation, recurrence, notice], commitAllowed: false }.
