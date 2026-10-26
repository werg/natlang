---
description: Construct a AccessibilityPackage draft for packageAccessibleMedia.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AccessibilityCheck[] }
returns: DraftAccessibilityPackage
---
Assemble the AccessibilityPackage as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: All deliverables align to the same approved media timeline and version.
