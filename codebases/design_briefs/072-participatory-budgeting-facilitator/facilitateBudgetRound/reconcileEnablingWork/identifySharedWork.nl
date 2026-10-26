---
description: Match common enabling tasks by activity identity and dependency evidence rather than similar wording.
args: { dossier: BudgetDossier; proposals: CivicProposal[] }
returns: EnablingWorkNode[]
---
Compare task identity, scope, location, beneficiary and dependencies in the evidence. Link one cost node only when proposals require the same work at the same scope. Similar descriptions without a common activity reference remain separate or unresolved. Preserve every proposal-to-work relation so revisions invalidate only dependent portfolios.
