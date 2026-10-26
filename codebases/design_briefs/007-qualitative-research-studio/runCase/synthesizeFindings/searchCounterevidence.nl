---
args: { codebook: CodeBoundary[], evidence: EvidencePage[] }
returns: CounterevidenceSearch
---
For each code boundary and each claimed mechanism actually supported by the corpus, call searchOneMechanism(code, mechanism, evidence). Consume each search's supporting and contradictory passage references, narrowed interpretation and missing partitions. Aggregate exact source coverage and keep claims scoped to searched sample windows. Return { supportingRefs, contradictoryRefs, searchedPartitions, missingPartitions, absenceClaimAllowed, codeMechanismSearches }. AbsenceClaimAllowed is true only when the declared corpus partitions were actually searched; no observed counterexample is not universal absence.
