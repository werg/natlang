---
args: { proposal: QualityEscapeProposal, receipt: QualityLotDispositionReceipt }
returns: QualityLotAssessment
---
For a release, require every receipt unit to appear in proposal.containment.releasableUnitIds and none in heldUnitIds; the lot must not be unresolved. Require proposal.effectiveness.closurePermitted true, a supported release basis for these units, and actual release verification references. If the physical system reports released when these gates fail, report safeToRelease false and identify the unsafe release for urgent containment; do not relabel it held. For a hold, require exact scoped held unit identities and physical-control evidence. Return each uncovered or unsafe unit.
