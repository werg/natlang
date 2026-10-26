---
description: Compare each verified qualification's authoritative validity interval with the exact mission window.
args: { mission: MissionDossier; change: MissionChange }
returns: VolunteerCapability[]
---
Use current issuing-authority evidence for qualification type, holder and valid-through date. Compare the validity interval to the mission date. Unknown or expired qualifications cannot cover safety-sensitive roles. Keep old credential evidence for audit and do not change capabilities unrelated to the expired credential.
