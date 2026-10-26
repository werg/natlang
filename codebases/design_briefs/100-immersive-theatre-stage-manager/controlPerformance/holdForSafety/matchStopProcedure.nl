---
description: Bind a received stop signal to its exact named production response and clearance authority.
args: { incident: ShowIncident; showEvidence: EvidenceItem[]; safetyAndConsentRules: EvidenceItem[] }
returns: StopProcedureMatch
---
Match the incident signal against current safety procedures. Return the exact procedure and named safety lead only when their source references are current and unambiguous. Preserve the supplied scope of the stop: scene, movement, venue zone, or full performance. An absent or conflicting procedure leaves the whole potentially affected activity held and names the missing authority; never invent a clearance chain.
