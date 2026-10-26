---
description: Recover a reference method, preserve methodological disagreement, and prepare reproducible tasks whose results are compared only from complete provenance-bound run receipts.
args: { request: ReplicationPlanRequest, evidence: ReplicationEvidenceBundle }
returns: ReplicationWorkbenchResult
uses: [replicationDesk/frameReplication, replicationDesk/recoverMethodProcedure, replicationDesk/constructDependencyGraph, replicationDesk/adaptUnavailableDependency, replicationDesk/preregisterSeedSensitivity, replicationDesk/pinDatasetVersion, replicationDesk/compileExecutablePlan, replicationDesk/verifyArtifactCompleteness, replicationDesk/compareCompletedRuns]
---
1. Bind frame = frameReplication(request, evidence.projects). Freeze target study, reproduction goal, source IDs, investigator, expected project version, and compute budget.
2. Bind procedure = recoverMethodProcedure(request, evidence.evidencePages, evidence.methodConflictEvidence). Preserve each step, parameter and source locator; prose/code contradictions remain competing interpretations for review.
3. Bind graph = constructDependencyGraph(procedure, evidence.dependencyReceipts). Order method steps by exact prerequisite, resource and source version; cycles and missing dependencies are blockers.
4. Bind adaptedGraph = adaptUnavailableDependency(graph, evidence.dependencyReceipts). A substitute dependency is labeled an adaptation and never replaces the original requirement or creates an exact replication claim.
5. Bind seeds = preregisterSeedSensitivity(request, evidence.seedPlanEvidence) before any outcome exists. When reference seed is absent, require the declared sensitivity set and forbid favorable-seed selection.
6. Bind dataset = pinDatasetVersion(request, evidence.datasetVersionReceipts). A changed content hash is an explicit adaptation; preserve expected and observed versions.
7. Only resolved method steps, dependencies, seed plan and dataset pin enter `executionPlan = compileExecutablePlan(request, procedure, adaptedGraph, seeds, dataset)`. Unresolved or ambiguous instructions remain blockers rather than guessed commands.
8. Bind artifactChecks = verifyArtifactCompleteness(executionPlan, evidence.jobReceipts). Partial crash outputs are not reusable unless every expected artifact, hash and provenance identity is verified.
9. Bind comparisons = compareCompletedRuns(executionPlan, evidence.jobReceipts, artifactChecks). Compare only complete runs with source, dataset, environment and seed bindings; keep incomplete jobs and adaptations separate.
10. Return { caseId: request.caseId, frame, procedure, dependencyGraph: adaptedGraph, seedPlan: seeds, datasetPin: dataset, executionPlan, artifactChecks, comparisons, adaptations: ["dependency substitute" when present, "dataset version mismatch" when present], blockers: procedure.unresolvedChoices + adaptedGraph.cycles + adaptedGraph.unavailableInputs + dataset.allowedAdaptation when nonempty, commitAllowed: false } as ReplicationWorkbenchResult. This plan does not claim execution without receipts.
