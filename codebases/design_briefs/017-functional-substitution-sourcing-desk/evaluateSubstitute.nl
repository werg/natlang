---
args: { request: SubstituteEvaluationRequest, catalog: SupplierCatalogSnapshot, prior?: SubstitutionCheckpoint|null }
returns: SubstitutionProgramResult
uses: [sourcingDesk/frameUse, sourcingDesk/selectSourcingPolicy, sourcingDesk/recoverFunctionalConstraints, sourcingDesk/generateMechanismCandidates, sourcingDesk/invalidateRevisionMatch, sourcingDesk/validateUnitConversion, sourcingDesk/extractRevisionPropertyClaims, sourcingDesk/compareCandidateRequirements, sourcingDesk/verifyLotAcceptance, sourcingDesk/designBoundedQualification, sourcingDesk/priceRequiredAdapter, sourcingDesk/scopeUseApproval, sourcingDesk/assembleUseBoundPacket]
---
Bind saved = prior when supplied, otherwise null.
1. Bind goal = frameUse(request, catalog.requests) and policy = selectSourcingPolicy(request, catalog.policyCandidates). Preserve intended function and approved procurement scope.
2. Bind requirements = recoverFunctionalConstraints(request, goal, catalog.evidencePages). If any critical requirement remains unknown, return the goal, policy, unresolved requirements, and a blocked proposal without ranking candidates.
3. Bind candidates = generateMechanismCandidates(request, requirements, catalog.evidencePages). Require supplier-specific item and current revision evidence.
4. Bind revision = invalidateRevisionMatch(request, catalog.revisionChanges, candidates). A silent revision change invalidates only findings tied to the superseded revision.
5. Bind conversion = validateUnitConversion(request, catalog.conversionReceipts). Preserve original units and use only exact conversion evidence.
6. Bind properties = extractRevisionPropertyClaims(candidates, catalog.evidencePages). Bind compatibility = compareCandidateRequirements(request, requirements, candidates, properties, conversion). Reject a candidate on any essential critical failure; do not rank unknown critical values as passing.
7. Bind lotRule = verifyLotAcceptance(request, catalog.lotAcceptanceRules). If sample representativeness is not established under the approved rule, keep production qualification blocked.
8. Bind qualification = designBoundedQualification(request, compatibility, catalog.evidencePages, catalog.lotAcceptanceRules). Set criteria and technical acceptance role before the supplier supplies a result.
9. Bind adapter = priceRequiredAdapter(request, catalog.adapterQuotes). Include adapter cost, lead time, and interface risk in total installed comparison.
10. Bind approval = scopeUseApproval(request, catalog.useApprovals). Reject approval reuse outside its exact machine, environment, and candidate revision.
11. Bind packet = assembleUseBoundPacket(request, candidates, compatibility, qualification, approval, adapter) only when all critical requirements pass and lot and approval scopes are settled.
12. Bind checkpoint = reconcileSubstitutionCheckpoint(saved, request, catalog, candidates, compatibility, qualification, packet). Preserve prior candidate revision lineage; lot acceptance and use approval apply only to the exact part revision, machine and environment. New supplier corrections stale only affected compatibility claims and dependent packet fields; duplicate qualification receipts are not new evidence.
13. Return { caseId: request.caseId, goal, policy, requirements, candidates, compatibility, qualification, packet, guardrails: [revision, conversion, lotRule, adapter, approval], checkpoint, commitAllowed: false }.
