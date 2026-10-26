---
description: Bind delivered equipment, accepted responsibility and completed transition steps to exact source receipts.
args: { dossier: TransitionDossier; change: TransitionChange; requirements: TransitionRequirement[]; participants: TransitionParticipant[] }
returns: PrerequisiteState[]
---
For each task prerequisite, require the source event that proves its completion: delivery scan for delivered equipment, explicit response for accepted care, and source evidence for completed logistics. Preserve a reservation as booked until delivery is observed. Leave any absent receipt as unknown and block only its dependent tasks.
