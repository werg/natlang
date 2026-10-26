# Research resource and experiment rescue coordinator

## Native plan shape

The entrypoint is `coordinateResearchWindow.nl`. It treats protocol requirements as hard predicates, resource/equipment/operator availability as a time-indexed constraint graph, and investigator preferences as a separate ranking layer. NatLang expands and ranks the finite assignment frontier; the only crisp scheduling primitive is `intervalsOverlap.ts`, which evaluates exact half-open interval arithmetic. NatLang decides which protocol predicates are essential and whether a proposed device is validated for the measurement.

The plan branches before scheduling. Missing protocol authority, sample timing, or operator qualification produces a blocked requirement rather than a guessed slot. A feasible slot is carried forward with its resource reservations. A calibration failure blocks readiness while the booking remains intact. A shrinking sample window re-solves hard constraints before optimizing calendar convenience. A changed-behavior substitute becomes a deviation proposal that cannot enter the ready branch until investigator approval is evidenced.

Cancellation is a separate reconciliation path. It preserves the prepared booking history, records consumed material and affected downstream experiments, and asks only the owners whose dependencies changed. The final result separates `slotProposal`, `rescueAlternatives`, `readiness`, `consumptionLedger`, and `reviewRequests`; neither a slot nor a readiness proposal starts an experiment.

## Regressions

The five exact fixture roots remain under the moved `coordinateResearchWindow/requirements|resources|slots|rescue|readiness/audit/qualify.nl` paths. Inputs and expected verdicts are unchanged. The native procedures do not infer a protocol deviation, operator qualification, or execution receipt from a calendar booking.

`reconcileResearchReservation.nl` is the return path for the external booking service. It matches a receipt against the exact experiment, instrument, operator and time interval selected by the plan; unknown, rejected and cancelled outcomes remain separate. Confirmed booking still does not assert day-of readiness or that an experiment ran.
