---
args: { receipt: ResearchReservationReceipt, assignments: WindowAssignment[] }
returns: ReservationProposalMatch
---
Find the proposed assignment for receipt.experimentId. The receipt matches only if exactly one assignment has the same experimentId, instrumentId, operatorId, startsAt, and endsAt, and the receipt has evidenceRefs. Do not treat an approximate window, substitute instrument, or different operator as the accepted reservation. Return every field mismatch and whether the proposed assignment is absent or duplicated.
