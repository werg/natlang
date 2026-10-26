---
description: Construct a PlainLanguageRelease draft for releaseAccessible.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ComprehensionProbe[] }
returns: DraftPlainLanguageRelease
---
Assemble the PlainLanguageRelease as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Independent fidelity review passes; readability alone is not the acceptance criterion.
