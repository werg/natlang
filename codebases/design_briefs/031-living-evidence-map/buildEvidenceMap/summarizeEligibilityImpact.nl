---
args: { original: ObservationLedger, rescreened: ObservationLedger, claims: EvidenceClaimGraph, changes: EvidencePage[] }
returns: ReviewUpdateLedger
---
Compare observations by canonical studyId and the exact outcome/evidence identity, preserving multiple measures from one study. Use the criteria-change pages to establish which records were reconsidered. A missing rescreened record is not automatically a retraction or exclusion; retain it as stale when its new eligibility disposition lacks evidence.

Bind rescreenedStudyIds to studies with evidenced rescreening. Bind staleStudyIds to studies whose changed criteria cannot yet be adjudicated. Follow actual claim-to-study support and opposition links to identify invalidatedClaimIds; a claim with lost or changed eligibility needs renewed applicability and confidence review even if other support survives. All other claim IDs enter preservedClaimIds. Set revisionReason to the actual criteria edition change, or to "No evidenced criteria change" when changes is empty. Set ownerQuestions to the missing screening evidence and affected claim review decisions. Return ReviewUpdateLedger { invalidatedClaimIds, preservedClaimIds, rescreenedStudyIds, staleStudyIds, revisionReason, ownerQuestions }. Never manufacture a new study result or change the old observation ledger.
