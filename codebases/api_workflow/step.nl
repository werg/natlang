import apply from "./step/apply";
import choose from "./step/choose";
import inspect from "./step/inspect";
---
description: Decide and apply one inventory, payment, shipping or recovery step.
args:
  acc: WorkflowState
  item: WorkflowEvent
returns: WorkflowState
---
function step(acc, item) -> WorkflowState
  current = inspect(acc.order_id)
  decision = choose(current, item)
  return apply(current, item, decision)
