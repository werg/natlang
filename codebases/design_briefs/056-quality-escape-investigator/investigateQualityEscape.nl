---
args: { caseFile: QualityEscapeCase }
returns: QualityEscapeProposal
---
1. Bind trace = traceReworkedLotExposure(caseFile.lotLineage, caseFile.processHistories). Follow every parent and rework branch; never replace an earlier route history with its latest route.
2. Bind comparison = compareInspectionRegimes(caseFile.observedDefects, caseFile.inspectionMethods, caseFile.evidence). If method coverage or thresholds differ, report detection change and mark rate comparison non-comparable; do not infer a defect-rate change from counts alone.
3. Bind causes = reviewCausalMechanisms(trace, comparison, caseFile.processHistories, caseFile.evidence). Treat operator identity only as provenance; no causal or blame claim follows from correlation.
4. Bind containment = reconcileUnitHoldScope(caseFile.holds, trace). Preserve unit-level holds and release only explicitly identified units with an applicable disposition.
5. Bind effectiveness = retestPriorClosure(caseFile.priorClosure, caseFile.correctiveActions, caseFile.observedDefects). If the defect recurs after closure, reopen the causal claim and require a new effectiveness check.
6. If effectiveness.reopened is true, keep closurePermitted false and preserve failed action targets. Return { caseId: caseFile.caseId, trace, comparison, causes, containment, effectiveness, commitAllowed: false }.
