---
args:
  observation: MarketObservation
returns: TradeIntent
---
Choose buy or pass for this merchant. Use only the merchant's own cash and
goods plus public offers. A buy names an offered seller and good and a positive
integer quantity. Consider price and likely future utility; do not assume
private inventories of other merchants. Exact settlement will reject
unavailable stock or insufficient cash.
