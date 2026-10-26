---
description: Construct a AccessibleSegment draft for writeAccessibleText.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AccessibilityChannelPlan[] }
returns: DraftAccessibleSegment
---
Assemble the AccessibleSegment as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Text is accurate, appropriately timed, and does not spoil future information.
