---
args: { graph: AnalysisGraphPlan, audit: AnalysisExecutionAudit, meaning: AnalysisMeaningMap, analysis: SemanticWorkOrderRequest }
returns: AnalysisAnswerProposal
---
State the question interpretation, materialized values, output grain, exclusions, and unresolved components. Check every number against the supplied execution evidence and every semantic qualifier against the selected meaning. If no actual materialized result is supplied, return only reproducible work-order steps and state that no result was computed.
