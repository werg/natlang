---
args: { candidate: RemedyBundle, decisions: DecisionsRecoveryAction, evidence: EvidencePage[] }
returns: DraftRecoveryAction
---
1. Construct an evidence-backed RecoveryAction record from established atoms, using the fields declared for RecoveryAction in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Sequence approved refunds, replacements, pickups, and communications. Reserve scarce replacements before promising dates, wait for required customer choices, and reconcile uncertain external outcomes rather than retrying irreversible operations blindly.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
