---
args: { revisions: FabricationRevision[], route: ShopRouteAssessment, offers: CurrentMaterialOffers }
returns: ToolingRevisionImpact
---
For each approved design revision that changes tooling, identify the affected setup and process route and which prior cost lines are stale. Update route and cost basis together; keep unapproved revision effects unresolved.
