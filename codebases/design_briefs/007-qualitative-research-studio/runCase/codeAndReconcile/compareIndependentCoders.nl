---
args: { passages: InterviewPassage[], codebook: CodeBoundary[], evidence: EvidencePage[] }
returns: CoderComparison[]
---
Compare independent labels on the prespecified overlap. Separate disagreement caused by unclear boundaries from a credible alternative interpretation. Preserve minority readings and bind each decision to the exact codebook version.
