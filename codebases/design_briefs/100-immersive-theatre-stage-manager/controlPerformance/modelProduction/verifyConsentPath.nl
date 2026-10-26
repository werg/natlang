---
description: Check each participation route against the participant's current choice and approved equivalent paths.
args: { incident: ShowIncident; scenes: TheatreScene[]; showEvidence: EvidenceItem[]; safetyAndConsentRules: EvidenceItem[] }
returns: ConsentAssessment
---
For every affected participant, use the latest current consent evidence for this exact scene and activity. A decline closes that participation path immediately. A quiet alternative counts as equivalent only when approved evidence shows it reaches the same dramatic outcome without another ask or a hidden penalty. If no alternative is documented, mark the path blocked and ask the stage manager for a nonparticipatory route. Never infer consent from attendance or an earlier scene.
