---
args:
  acc: State
  item: Event
returns: State
types:
  Event: '{ id: Text, order: Text, text: Text }'
  Kind: '"start" | "paid" | "failed" | "sent" | "cancel"'
  Order: '"reserved" | "paid" | "done" | "cancelled"'
  Command: '{ key: Text, order: Text, operation: Text }'
  State: '{ seen: Text[], orders: Dict<Order>, outbox: Command[] }'
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
