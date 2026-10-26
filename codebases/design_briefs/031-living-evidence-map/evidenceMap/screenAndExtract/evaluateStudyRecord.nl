---
args: { criteria: EligibilitySnapshot, lineage: StudyLineage, report: Evidence }
returns: ExtractedObservation
---
Apply the frozen inclusion and exclusion criteria to this report edition. Distinguish a report version from its linked study identity. For an eligible record, extract only reported design, population, outcome, value, original unit, follow-up and limitations, with exact evidence references. If the report is excluded, preserve its study identity and the criterion that decided exclusion. Abstract-only access is a limit, not permission to infer missing methods or results.
