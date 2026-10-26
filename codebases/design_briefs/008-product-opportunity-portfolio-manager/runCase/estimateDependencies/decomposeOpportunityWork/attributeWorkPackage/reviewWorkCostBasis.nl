---
args: { intervention: InterventionMechanism, candidatePackage: WorkPackage, evidence: EvidencePage[] }
returns: CostBasisReview
---
Match every estimate to this intervention, work package, time horizon and source version. Distinguish quoted range from forecast, internal effort from external spend, and funded capacity from an assumption. Return eligibleForReceipt only when the exact package scope and estimate basis are established; otherwise list the missing or conflicting evidence and preserve the supplied range without inventing a total.
