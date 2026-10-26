---
args: { output: ApprovedAnalysisOutput, searches: EvidencePage[] }
returns: SubgroupSearchReview
---
For each supplied subgroup search call classifySearchTiming(output, search). Use the actual timing relative to approved analysis and declared subgroup plan. Compare observed searches with the output's stated search count; missing history remains unknown, not zero. Return prespecified only when all relevant searches were approved in advance; otherwise preserve exploratory status and exact missing search records.
