---
args: { revisions: LocalDrawingRevision[], evidence: Evidence[] }
returns: ScopePartition
---
Bind findings = revisions.map(revision => assessDrawingRevision(revision, evidence)). Preserve each finding by drawing and revision. Partition affectedAreas from findings whose changed geometry or function is material; partition unaffectedAreas from preservedAreas. Set documentationOnly only when every revision finding is documentation-only. Return unresolvedAreas for any revision lacking evidence needed to distinguish reference-only edits from changed scope.
