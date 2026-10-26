---
description: Reframe narrative claims after artifact availability or evidence changes, without silently substituting objects.
args: { dossier: ExhibitionDossier; change: ExhibitionChange; constraints: ObjectConstraint[] }
returns: NarrativeSupport[]
---
Call `traceArtifactSupport(dossier, change, constraints)` for each argument claim. Remove a withdrawn artifact from active support. If the claim loses its evidence, narrow or remove the claim; introduce another object only through a new argument review demonstrating that it supports the same stated claim. Preserve competing interpretation and provenance qualifications.
