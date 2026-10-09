---
description: A merchant's policy for the tick. Choose to buy one offered good from one other merchant, or to pass, from what the merchant knows.
args:
  observation: MarketObservation
returns: TradeIntent
---
Choose the intent of the merchant observation.actor for this tick. Everything the merchant knows is in observation: its
own cash and goods, and the public offers of the other merchants (seller, good, price per unit).

1. List the offers whose price times some quantity of at least 1 the merchant can pay with observation.cash.
2. Weigh each such offer by what the price buys against the merchant's likely future use of the good, and against
   the goods it already holds.
3. When one offer is worth taking, return a buy: kind "buy", seller and good exactly as the offer names them, and a
   quantity that cash covers. Otherwise return kind "pass".
