---
args: {"promises": "PromiseEpisode[]", "progress": "PromiseProgress[]", "docket": "DepartureDocket"}
returns: "ContinuityPrerequisite[]"
---
Expand each remaining obligation into specific knowledge, access, approval and external-relationship dependencies. Bind edges = traceBlockedHandoff(promise, progress, docket.correspondence). Collect only capabilities needed by the current uncompleted scope. Credential values must never enter a dependency or dossier; use an authorized secret reference or a minimal access request. Each blocker needs an owner role, recovery task and receipt requirement. Cycles stay explicit with the action needed to break them.
