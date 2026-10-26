---
args: { materials: MaterialOffer[], revisions: FabricationRevision[], route: ShopRouteAssessment }
returns: MaterialPriceReview
---
Bind current = separateCurrentMaterialOffers(materials, route). Bind tooling = reconcileToolingRevisionImpact(revisions, route, current). Use only compatible, unexpired prices for the quote date. A design revision affecting tooling changes both route and cost; stale pricing cannot support a committed estimate.
