---
args: { revision: LocalDrawingRevision, evidence: Evidence[] }
returns: DrawingRevisionFinding
---
1. Preserve drawingId, revision, and evidenceRef as the source identity. If evidenceRef is absent, identify changed and preserved areas as unresolved instead of treating declared labels as verified geometry.
2. When referenceOnly is true, compare the supplied changed and unchanged area lists; treat the revision as documentation-only only when no functional area change is declared and evidence supports unchanged function and geometry.
3. For substantive revisions, carry only revision.changedAreas into affectedAreas and revision.unchangedAreas into preservedAreas. An area listed in both sets remains unresolved until the drawing owner resolves the conflict.
4. Return every overlap, missing evidence reference, or contradictory area claim in unresolvedAreas.
