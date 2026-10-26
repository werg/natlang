---
description: Bind named transition participants to current role acceptance and consent window.
args: { dossier: TransitionDossier; change: TransitionChange; requirements: TransitionRequirement[] }
returns: TransitionParticipant[]
---
For each responsible role, require the named participant's current acceptance for the exact task and time window. Preserve caregiver inability or nonresponse as unaccepted; escalate the responsibility rather than assuming availability. Keep the person's consent scope distinct from their general availability.
