---
args: { tender: TenderComparisonDossier }
returns: TenderComparison
---
1. Bind cells = classifyBidCoverage(tender.bidScopes, tender.requirements, tender.bidRevisions). A price allowance remains uncertain and is not a firm quantity.
2. Bind seams = assignTradeInterfaces(tender.interfacePairs, cells). If two trades each assume the other supplies a connection, create one explicit acceptance task with an unresolved owner.
3. Bind amended = applyTenderClarifications(cells, tender.clarifications, tender.bidRevisions). A late revision removing an inclusion invalidates the prior coverage cell and every clarification result derived from it.
4. Bind omissions = checkCommissioningCoverage(amended, tender.commissioningRequirements). A lower total with omitted commissioning is functionally incomplete and must not rank as complete.
5. Return { cells: amended.cells, seams, uncertainAllowanceIds: amended.uncertainAllowanceIds, functionalOmissions: omissions, invalidatedCellIds: amended.invalidatedCellIds, rankable: omissions.length == 0 && seams.length == 0 && amended.blockers.length == 0, blockers: amended.blockers }.
