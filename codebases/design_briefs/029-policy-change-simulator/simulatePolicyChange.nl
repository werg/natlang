---
description: Compare exact, version-bound policy execution artifacts across a frozen case cohort, preserving indeterminate cases and rare-safeguard regressions.
args: { request: PolicyScenarioRequest, dossier: PolicySimulationDossier }
returns: PolicySimulationResult
uses: [policySimulator/frameSimulation, policySimulator/compilePolicyDelta, policySimulator/forkEditedSnapshot, policySimulator/freezeCaseCohort, policySimulator/bindPolicyReplayEvidence, policySimulator/compareExactOutcomes, policySimulator/protectExceptionRegression, policySimulator/traceBoundaryCases, policySimulator/proposePolicyOption, policySimulator/bindAdoptedVersion]
---
Bind saved = dossier.checkpoint when present, otherwise null. An absent checkpoint means this is the first review.

Initialize outcomes = null, regressions = null, boundaries = null and proposal = null.

1. Bind goal = frameSimulation(request, dossier.simulations). Freeze the two policy versions and exact case collection before work begins.
2. Bind delta = compilePolicyDelta(request, dossier.policyCandidates). Resolve the authority and effective scope of each version and describe only condition-linked changes.
3. Bind editFork = forkEditedSnapshot(request, dossier.ownerEdits). If the candidate changed during evaluation, retain the old snapshot and fork a new run; do not compare runs across candidate hashes.
4. Bind cohort = freezeCaseCohort(request, dossier.cases, dossier.dedupChanges). Preserve exact case IDs and sources, mark missing required fields indeterminate, and separate cohort dedup changes from outcomes.
5. Bind replayBinding = bindPolicyReplayEvidence(request, cohort, dossier.replayPairs). Require a matching baseline/candidate receipt for each comparable case, same input digest and execution configuration, distinct run IDs, and exact policy hashes.
6. Only if candidate hash remains stable and at least one exact pair is verified, bind outcomes = compareExactOutcomes(delta, cohort, replayBinding). This function consumes exact supplied run artifacts; it never executes or predicts a policy.
7. Only after pairwise comparison, bind regressions = protectExceptionRegression(outcomes, dossier.evidencePages). Inspect rare safeguards individually and preserve every regression case before considering aggregate results.
8. If comparable pairs and regression tests are complete, bind boundaries = traceBoundaryCases(outcomes, regressions, cohort). Keep untested combinations untested and missing-field cases indeterminate.
9. Bind adoption = bindAdoptedVersion(request, dossier.adoptionReceipts). Adoption must reference the tested candidate hash; a later edit requires a new fork and complete replay.
10. When boundaries are complete and owner review is clear, bind proposal = proposePolicyOption(request, delta, boundaries, regressions). Otherwise proposal is null.
11. Bind checkpoint = reconcilePolicyScenario(saved, request, dossier, delta, cohort, replayBinding, outcomes, regressions, boundaries, adoption). Preserve frozen cohort membership and exact baseline/candidate hashes. Candidate edits fork a new run and stale only replay pairs on the edited hash. Keep indeterminate cases and rare-safeguard failures; adoption receipt must cite the tested candidate hash.
12. Return { caseId: request.caseId, delta, cohort, replayBinding, outcomes, regressions, boundaries, adoption, proposal, openQuestions: replayBinding.issues, checkpoint, commitAllowed: false } as PolicySimulationResult.
