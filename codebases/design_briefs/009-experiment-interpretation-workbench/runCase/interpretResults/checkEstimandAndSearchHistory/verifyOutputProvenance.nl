---
args: { output: ApprovedAnalysisOutput, design: EffectiveDesignTimeline[], approvals: EvidencePage[] }
returns: AnalysisProvenanceReview
---
Bind outputReceipt = matchAnalysisReceipt(output, design, approvals). Require exact analysis ID, snapshot ID, effective design version, approval reference and source identity. A later approval or same-valued result from another snapshot does not bind this output. Return each match flag, the exact source references and every missing or conflicting provenance item.
