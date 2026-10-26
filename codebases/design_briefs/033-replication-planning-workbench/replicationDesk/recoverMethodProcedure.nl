---
args: { request: ReplicationPlanRequest, pages: EvidencePage[], conflicts: EvidencePage[] }
returns: MethodProcedure
---
Reconstruct the reference method as ordered, source-located steps with inputs, parameters, versions, expected artifacts, and decision points. When paper prose and executable code disagree, preserve both interpretations and bind the disagreement to exact sources; do not silently treat code as authoritative or repair the paper from memory.
