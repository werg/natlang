---
description: Construct a AccessibilityChannelPlan draft for planChannels.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: NarrativeInformation[] }
returns: DraftAccessibilityChannelPlan
---
Assemble the AccessibilityChannelPlan as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Channel plan preserves essential information within actual timing and capacity constraints.
