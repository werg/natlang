---
args: { dossier: ChangeOrderDossier }
returns: ChangeOrderImpactPacket
---
1. Bind authority = reconcileInstructionAuthority(dossier.instruction, dossier.writtenConfirmations, dossier.fieldReports). Preserve a disputed verbal instruction and contemporaneous event separately; work performed before commercial approval is an event record, not retroactive authorization.
2. Bind scope = isolateChangedDrawingAreas(dossier.drawingRevisions). A drawing revision limited to one area cannot flow into unchanged areas unless an explicit dependency supports it.
3. Bind dependencies = traceChangeDependencies(scope, dossier.dependencies). Return affected work packages and retain unaffected scope.
4. Bind schedule = testAvailableFloat(dossier.schedule, dependencies). Compare the changed critical path with the supplied activity durations and float; state assumptions and missing predecessor data. Do not assert delay where available float absorbs the change.
5. Bind costs = removeDuplicateChangeAttribution(dossier.costItems, dossier.fieldReports). Attribute each supported work item once across overlapping changes; preserve actual cost evidence even if approval is disputed.
6. Return { caseId: dossier.caseId, authority, affectedAreas: scope.affectedAreas, unaffectedAreas: scope.unaffectedAreas, dependencies, floatAnalysis: schedule.findings, nonDuplicatedCostItems: costs.acceptedIds, excludedDuplicateCostIds: costs.duplicateIds, proposalStatus: authority.confirmed && schedule.blockers.length == 0 ? "review-required" : "needs-review", commitAllowed: false }.
