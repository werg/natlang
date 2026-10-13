---
description: Build complete contingent candidates by a finite obligation frontier and choose the least costly valid policy.
args:
  context: SearchContext
returns: RepairPolicy
---
1. If required obligation facts or observation choices remain unresolved, return a blocked RepairPolicy with the
   context's obligations and observations. Preserve why a sound complete policy cannot yet be selected.
2. Form the Cartesian product of selected query answer outcomes, retaining only source-supported combinations.
   Each PolicyBranch has its query/answer/ledger/status conditions and initially no actions. With no queries,
   use one branch with empty conditions. This product is finite because all outcomes are source-listed.
3. Create a single empty PolicyCandidate with those branches, no covered obligations or common actions,
   zero writes and observationCost equal to the distinct selected query costs.
4. Initialize RepairSearchState with all obligation IDs remaining and that candidate. Call
   expandFrontier.iterateOn(initialState) with measure state.remainingObligationIds.length until it is zero.
   Each expansion decides the next obligation for every surviving candidate; it never re-adds an obligation.
5. Consume the final candidates. Remove candidates that fail to cover every obligation or leave any branch
   without its required entitlements. Keep rejection reasons as explanations of the eliminated alternatives.
6. Compute each candidate's worstCaseWrites as commonActions.length plus the maximum branch action count.
   Minimize that number; then minimize total observationCost. Equivalent independent action order is not a cost.
7. When several candidates tie, return any one along with its source-backed rejected alternatives. Do not make
   generated action IDs, independent action order or branch enumeration order part of policy correctness.
8. Return the winning candidate with observations and obligations; if none survives, return null candidate and
   accumulated blocking reasons. No fixed search budget truncates the finite obligation frontier.
