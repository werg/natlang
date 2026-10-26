---
args: { materials: MaterialOffer[], revisions: FabricationRevision[], route: ShopRouteAssessment }
returns: MaterialPriceReview
---
Include only material offers compatible with the selected process and valid on the quote date. An expired offer makes pricing provisional. When a customer-approved revision affects tooling, identify the changed route and tooling cost basis together; never carry forward a superseded price as current.
