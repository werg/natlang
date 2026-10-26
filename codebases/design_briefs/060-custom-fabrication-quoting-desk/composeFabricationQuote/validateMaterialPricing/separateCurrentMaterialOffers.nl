---
args: { materials: MaterialOffer[], route: ShopRouteAssessment }
returns: CurrentMaterialOffers
---
Keep only materials compatible with the selected process and whose validity period covers the quote date. Preserve expired and incompatible offer IDs separately; do not silently extend a supplier price.
