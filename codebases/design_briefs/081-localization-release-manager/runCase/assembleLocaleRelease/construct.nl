---
description: Construct a LocaleReleasePacket draft for assembleLocaleRelease.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: LocaleQualityFinding[] }
returns: DraftLocaleReleasePacket
---
Assemble the LocaleReleasePacket as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Bundle is internally version-consistent and every fallback is declared and safe.
