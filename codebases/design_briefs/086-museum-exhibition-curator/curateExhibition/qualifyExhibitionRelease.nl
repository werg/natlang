---
description: Qualify the complete exhibition release against conservation limits, permissions, label evidence and all derivatives.
args: { dossier: ExhibitionDossier; change: ExhibitionChange; constraints: ObjectConstraint[]; arguments: NarrativeSupport[]; zones: VisitorContextEdge[]; labels: ExhibitionLabel[] }
returns: ExhibitionRelease
---
Call `reconcilePlacementLimits(dossier, change, constraints)` for each artifact and physical zone. Check actual light exposure and duration against the latest conservation report; changed conditions invalidate affected placements and dependent labels. Confirm community restrictions cover all online and physical derivatives, and all visitor-path tests preserve needed context. Release only after every object permission and label qualification is current.
