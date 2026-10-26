---
description: Construct a CommissionBrief draft for commissionPieces.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: IssueCoverage[] }
returns: DraftCommissionBrief
---
Assemble the CommissionBrief as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Contributor accepts the exact scope and commercial conditions before work is committed.
