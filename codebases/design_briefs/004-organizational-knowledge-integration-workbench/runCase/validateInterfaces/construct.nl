---
args: { candidate: IntegrationWave, decisions: DecisionsInterfaceScenario, evidence: EvidencePage[] }
returns: DraftInterfaceScenario
---
1. Construct an evidence-backed InterfaceScenario record from established atoms, using the fields declared for InterfaceScenario in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Replay cross-team cases using proposed terminology and responsibilities. Look for steps both teams assume the other performs; turn failures into interface decisions and rerun only affected scenarios before approving a migration wave.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
