---
args: { dossier: ReuseFeasibilityDossier }
returns: ReuseFeasibilityReview
---
1. Bind evidence = reconcileSurveyWithHistoricPlans(dossier.siteFacts, dossier.historicDrawings, dossier.surveys). Preserve every drawing and survey; use a qualified, current survey for present conditions while exposing conflict and uncertainty.
2. Bind constraints = recheckConstraintsForCurrentUse(dossier.intendedUses, dossier.constraints, evidence). A changed intended use invalidates only constraints and investigations dependent on that use.
3. Bind assumptions = gateStructuralAssumptions(dossier.structuralAssumptions, dossier.concepts). A concept resting on an untested structural claim remains contingent until qualified review resolves it.
4. Bind enabling = auditEnablingWorks(dossier.concepts, dossier.costScopes). Expose enabling items omitted from the cost estimate rather than comparing incomplete totals.
5. Bind circulation = validatePhasedPublicRoutes(dossier.phases, dossier.circulationRoutes). A public opening cannot share a conflicting temporary route through active construction without a resolved access plan.
6. Return { caseId: dossier.caseId, evidence, affectedConstraints: constraints.affectedIds, contingentOptions: assumptions.contingentOptionIds, investigationGates: assumptions.reviewTasks, missingEnablingWorks: enabling, phaseAccessConflicts: circulation, optionDisposition: constraints.blockers.length > 0 || circulation.length > 0 ? "needs-review" : "feasibility-comparison", commitAllowed: false }.
