---
description: Require current logistics readiness evidence and receiving-organization acknowledgment before handoff is ready.
args: { dossier: TransitionDossier; change: TransitionChange; taskGraph: PrerequisiteState[]; schedule: ScheduleImpact[] }
returns: TransitionReadiness[]
---
Call `identifyClinicalEscalations(dossier, change, taskGraph)` for substitutions that touch clinical requirements. Require current clinician review before such a change can proceed. Independently require each dependency receipt and a receiving-organization acknowledgment bound to the current plan version. A prepared plan without acknowledgment remains a blocked handoff.
