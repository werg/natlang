import { clear_outbox } from "./step/clear_outbox";
import { dispatch } from "./step/dispatch";
import { read_event } from "./step/read_event";
import { seen } from "./step/seen";
import { transition } from "./step/transition";
---
args:
  acc: State
  item: Event
returns: State
types:
  Event: '{ id: string, order: string, text: string }'
  Kind: '"start" | "paid" | "failed" | "sent" | "cancel"'
  Order: '"reserved" | "paid" | "done" | "cancelled"'
  Command: '{ key: string, order: string, operation: string }'
  State: '{ seen: string[], orders: Record<string, Order>, outbox: Command[] }'
effects:
- queue.send
description: Process an event stream with duplicate suppression, an outbox, idempotent
  delivery and compensation.
---
function step(acc, item) -> State
  duplicate = seen(acc, item)
  if duplicate:
    return acc
  else:
    kind = read_event(item.text)
    next = transition(acc, item, kind)
    sent = dispatch(next.outbox)
    if dispatch quiesced: resume that same dispatch call once
    if dispatch still quiesced: report a blocker and preserve the pending work
    return clear_outbox(next)
