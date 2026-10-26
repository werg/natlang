---
args: { mandates: ProtocolMandate[], snapshot: ResearchOperationsSnapshot }
returns: InstrumentReview
---
Compare the approved mandate capability and validated-use records with each available instrument's capabilities, current calibration, maintenance, and measured behavior. Separate truly eligible instruments from behavior-different substitutes. Preserve booked devices with an expired calibration in bookedButUnavailable; availability does not cancel the booking.
