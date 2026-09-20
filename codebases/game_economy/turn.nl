---
description: Choose one merchant trade or pass from its permitted observation.
args:
  actor: Text
returns: TradeReceipt
---
function turn(actor) -> TradeReceipt
  observation = observe(actor)
  intent = decide(observation)
  return submit(actor, observation.tick, intent)
