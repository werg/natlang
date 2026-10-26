---
args: { meaning: AnalysisMeaningMap, support: FieldSupportMap, analysis: SemanticWorkOrderRequest }
returns: AnalysisGraphPlan
---
Build nodes for authorized filters, temporal matching, semantic judgments, joins, and aggregation. Attach source lineage and missing-data behavior to each edge. Before accepting a join, compare its declared cardinality with the output grain; do not permit one-to-many expansion to silently inflate counts. Check graph cycles before returning the plan.
For each join bind grainCheck = checkJoinCardinality(join, meaning.outputGrain). Bind typeCheck = checkAcyclicTypedGraph(nodes, edges); any failed check blocks materialization.
