---
args: { candidate: CodeBoundary, sample: SamplingManifest, pilot: EvidencePage[] }
returns: CodeBoundaryReview
---
For each passage in the frozen selected sample that the candidate cites or that could distinguish its meaning, bind judgment = assessBoundaryExample(candidate, passage). Retain the actual observed wording and passage identity. Group examples by include, exclude, counterexample, or ambiguous role. If inclusion and exclusion examples collapse under the proposed boundary, mark disposition overlapping and retain the competing meaning; if the pilot lacks enough contrast, mark unresolved. Return the revised boundary, every judgment, its disposition, a concise evidence-based rationale, and the precise boundary questions.
