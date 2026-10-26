---
args: { codebook: CodeBoundary[], evidence: EvidencePage[] }
returns: CounterevidenceSearch
---
For every code boundary and claimed mechanism, search the declared corpus for passages that contradict or narrow it. Preserve search coverage and missing partitions. If no counterexample appears, report that none was observed in the searched sample rather than claiming universal absence.
