---
description: Construct a AccessibilityCheck draft for validateSynchronization.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AccessibleSegment[] }
returns: DraftAccessibilityCheck
---
Assemble the AccessibilityCheck as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact synchronization passes and essential information remains understandable across channels.
