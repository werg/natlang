---
args: { candidate: ChangeConsequence, decisions: DecisionsChangeOption, evidence: EvidencePage[] }
returns: DraftChangeOption
---
1. Construct an evidence-backed ChangeOption record from established atoms, using the fields declared for ChangeOption in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Form a small set of permitted choices centered on the requested outcome. Include the direct exit path without manipulative friction, disclose linked-service effects, and avoid presenting an unrequested alternative as the default selection.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
