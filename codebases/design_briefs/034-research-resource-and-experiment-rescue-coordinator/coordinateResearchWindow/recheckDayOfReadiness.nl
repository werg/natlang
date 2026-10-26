---
args: { assignments: AssignmentSearch, mandates: ProtocolMandate[], snapshot: ResearchOperationsSnapshot }
returns: ReadinessDecision[]
---
1. Bind equipment = verifyInstrumentStatus(assignments, snapshot); current booking does not establish calibration or maintenance clearance.
2. Bind biology = verifySampleViability(assignments, mandates, snapshot); a shrunk usable window can invalidate the proposed start without altering the booking history.
3. Bind staffing = verifyOperatorPresence(assignments, snapshot) and deviations = verifyDeviationApproval(assignments, snapshot).
4. Merge these evidence reviews by experiment. Any failed calibration, sample interval, or operator requirement makes status blocked; an unapproved behavior change is review; only current positive evidence for every required condition can be ready. Always keep bookingRetained true for a booked experiment.
