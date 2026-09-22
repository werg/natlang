import { choose } from "./turn/choose";
import { observe } from "./turn/observe";
import { submit } from "./turn/submit";
---
description: Choose a fighter tactic at a round boundary.
args:
  actor: string
returns: CombatReceipt
---
function turn(actor) -> CombatReceipt
  perception = observe(actor)
  plan = choose(perception)
  return submit(actor, perception.round, plan)
