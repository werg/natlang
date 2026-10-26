---
args: { mandates: ProtocolMandate[], assignments: AssignmentSearch, snapshot: ResearchOperationsSnapshot }
returns: RescueAlternative[]
---
For each unscheduled experiment test a later sample window only when an approved protocol version covers it. If the current protocol does not cover the later interval, return a postponement question to the protocol owner and no new assignment. Preserve current sample viability limits and the current failed scheduling result.
