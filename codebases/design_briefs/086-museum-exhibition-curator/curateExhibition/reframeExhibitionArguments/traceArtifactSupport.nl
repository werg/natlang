---
description: Trace each exhibition claim to current artifact evidence and loan availability.
args: { dossier: ExhibitionDossier; change: ExhibitionChange; constraints: ObjectConstraint[] }
returns: NarrativeSupport[]
---
For each claim, list the specific evidence-bearing artifacts. Separate current, restricted, withdrawn and uncertain objects. Retain a claim only when current evidence supports it at the proposed scope. Do not treat a thematically similar artifact as equivalent evidence.
