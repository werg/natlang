---
args: { analysis: SemanticWorkOrderRequest }
returns: SemanticWorkOrderProposal
---
1. Bind meaning = decomposeQuestionSemantics(analysis). Every inclusion, time, population, and aggregation phrase gets an explicit interpretation; a decisive missing time window becomes a question.
2. Bind suitability = bindMeaningToAuthorizedFields(meaning, analysis). Exclude inaccessible records before semantic model exposure and distinguish unavailable data from restricted data.
3. Bind graph = compileGrainPreservingAnalysisGraph(meaning, suitability, analysis). Declare every join's expected cardinality and ensure its output grain matches the question.
4. Bind execution = reconcileAnalysisPartitions(graph, analysis). If a prior run used another rubric version, keep that run pinned and propose targeted reclassification. If only grouping changed, reuse the qualified row judgments and recompute the aggregate.
5. Bind answer = explainMaterializedAnalysis(graph, execution, meaning, analysis). Cite every numeric claim to materialized result evidence; if execution evidence is absent, return a work order without invented values.
6. Return { requestId: analysis.requestId, meaning, suitability, graph, execution, answer, commitAllowed: false }.
