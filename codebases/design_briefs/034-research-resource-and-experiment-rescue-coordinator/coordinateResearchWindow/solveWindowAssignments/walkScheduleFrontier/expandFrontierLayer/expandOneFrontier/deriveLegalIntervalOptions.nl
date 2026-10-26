---
args: { experimentId: string, mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot, selected: WindowAssignment[] }
returns: StartWindowOptions
---
For each eligible instrument derive start candidates from its availability start, the sample window start, and the end of a conflicting selected or booked interval. Keep only starts whose duration ends before both the protocol sample deadline and resource-window end. The interval-boundary set is finite; preserve all distinct alternatives for downstream exact checks.
