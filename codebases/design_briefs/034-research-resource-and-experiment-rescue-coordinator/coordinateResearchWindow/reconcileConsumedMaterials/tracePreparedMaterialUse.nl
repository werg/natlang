---
args: { snapshot: ResearchOperationsSnapshot, assignments: AssignmentSearch }
returns: MaterialTrace
---
Match each cancellation to preparation records using both experiment ID and material ID. Distinguish prepared, consumed, reusable, disposed, and unknown amounts. Preserve exact event time and evidence version; do not convert an unmeasured quantity into zero consumption.
