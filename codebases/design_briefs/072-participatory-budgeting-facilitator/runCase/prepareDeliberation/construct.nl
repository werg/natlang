---
description: Construct a BudgetDecisionPacket draft for prepareDeliberation.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CivicPortfolio[] }
returns: DraftBudgetDecisionPacket
---
Assemble the BudgetDecisionPacket as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Selection follows the published process and the selected portfolio matches its approved scope.
