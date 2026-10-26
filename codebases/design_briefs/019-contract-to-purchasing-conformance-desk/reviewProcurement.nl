---
args: { request: ProcurementReviewRequest, agreementFile: AgreementDeskFile }
returns: ProcurementProgramResult
uses: [procurementDesk/frameNeed, procurementDesk/selectCurrentProcurementPolicy, procurementDesk/pinCatalogVersion, procurementDesk/recheckExpiredTerms, procurementDesk/aggregateCommitments, procurementDesk/verifyEntityCoverage, procurementDesk/repriceBundleRemoval, procurementDesk/interpretPurchaseNeed, procurementDesk/scopeAgreementToLegalEntity, procurementDesk/compareLineTerms, procurementDesk/constructCompliantOptions, procurementDesk/routeCumulativeApproval]
---
1. Bind catalogPin = pinCatalogVersion(request, agreementFile.catalogVersions). Bind expiry = recheckExpiredTerms(request, agreementFile.waitingApprovals) before using a pending approval; expired terms require refreshed price and scope evidence.
2. Bind goal = frameNeed(request, agreementFile.purchases). Bind policy = selectCurrentProcurementPolicy(request, agreementFile.policyCandidates).
3. Bind intent = interpretPurchaseNeed(request, agreementFile.purchases, agreementFile.evidencePages). If essential quantity, purpose, or purchasing entity remains unknown, return unresolved intent without selecting a supplier item.
4. Bind entity = verifyEntityCoverage(request, agreementFile.entityCoverageTerms). Bind agreement = scopeAgreementToLegalEntity(request, intent, agreementFile.evidencePages, agreementFile.entityCoverageTerms). No conformance finding is accepted if the affiliate scope is unsupported.
5. Bind conformance = compareLineTerms(intent, agreement, agreementFile.bundlePriceSchedules, agreementFile.catalogVersions). Keep function coverage distinct from price compliance.
6. Bind commitments = aggregateCommitments(request, agreementFile.relatedCommitments) before approval routing. Bind repricing = repriceBundleRemoval(request, agreementFile.bundlePriceSchedules) whenever an item is removed from a discount bundle.
7. Bind options = constructCompliantOptions(intent, agreement, conformance, repricing) only when essential need and agreement scope are supported; retain exception route otherwise.
8. Bind disposition = routeCumulativeApproval(request, options, commitments, agreement, expiry, policy). A pending order must revalidate terms before release and cannot bypass thresholds through multiple requests.
9. Return { caseId: request.caseId, goal, policy, intent, agreement, conformance, options, disposition, guardrails: [catalogPin, expiry, commitments, entity, repricing], commitAllowed: false }.
