---
args: { mandates: ProtocolMandate[], snapshot: ResearchOperationsSnapshot }
returns: ReservationReview
---
Overlay booked, preparation, maintenance, and shared-person intervals. Treat one person as one constrained resource across projects. Return exact unavailable intervals and booking references, including any conflict where the instrument is free but its required operator is not.
