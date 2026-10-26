---
args: { passages: InterviewPassage[], codebook: CodeBoundary[], evidence: EvidencePage[] }
returns: CoderComparison[]
---
For each exact prespecified coder-overlap passage and current code version, compare only the source-backed labels belonging to those independent coders. For each disagreement call reviewOneCoderDisagreement(passage, code, evidence); consume its judgment to separate unclear boundaries, translation effects, credible minority interpretation, and insufficient records. Return one CoderComparison per passage/code version with agreements, disagreements, the child reviews, retained minority interpretations and evidence identities. Mark a comparison stale if its passage or code version is superseded; never convert unresolved disagreement into consensus.
