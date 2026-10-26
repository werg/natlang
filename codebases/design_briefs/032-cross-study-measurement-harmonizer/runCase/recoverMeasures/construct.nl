---
args: { candidate: Harmonization, decisions: DecisionsMeasureDefinition, evidence: EvidencePage[] }
returns: DraftMeasureDefinition
---
1. Construct an evidence-backed MeasureDefinition record from established atoms, using the fields declared for MeasureDefinition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read dictionaries, questionnaires, protocols, and usage notes together to recover what each variable measures. Distinguish similar labels from equivalent constructs and record timing, respondent, and collection-method differences that could change interpretation.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
