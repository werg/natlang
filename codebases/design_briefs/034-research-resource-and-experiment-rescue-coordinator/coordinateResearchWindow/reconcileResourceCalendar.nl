---
args: { mandates: ProtocolMandate[], snapshot: ResearchOperationsSnapshot }
returns: ResourceAssessment[]
---
1. Bind instrumentReview = assessInstrumentMatch(mandates, snapshot); exclude instruments with expired calibration, unvalidated measurement behavior, or incompatible maintenance status.
2. Bind operatorReview = verifyOperatorQualifications(mandates, snapshot); the calendar owner is not automatically an authorized experiment operator.
3. Bind reservationReview = examineSharedReservations(mandates, snapshot); compare the full interval against all existing projects and the same operator's other assignments.
4. Join the reviews by experiment ID into one ResourceAssessment per mandate. Keep booked-but-ineligible devices in blockedBy without erasing their reservation. Put a behavior-different near match in substituteReviewIds only; it remains an investigator decision.
