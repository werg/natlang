---
args: { graphNodes: AnalysisNode[], graphEdges: AnalysisEdge[] }
returns: GraphTypeCheck
---
Walk prerequisites from source nodes toward aggregation. Reject an edge whose output type is not accepted by its consumer, and reject a graph where unresolved nodes remain after all prerequisite-ready nodes are consumed. Preserve the cycle members and type mismatch at the failed edge.
