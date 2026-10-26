---
description: Construct a ComprehensionProbe draft for challengeComprehension.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PlainLanguageDraft[] }
returns: DraftComprehensionProbe
---
Assemble the ComprehensionProbe as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Source and rewrite imply the same consequential answers.
