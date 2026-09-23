import clear_outbox from "./step/clear_outbox";
import dispatch from "./step/dispatch";
import read_event from "./step/read_event";
import seen from "./step/seen";
import transition from "./step/transition";
---
args:
  acc: State
  item: Event
returns: State
effects:
  - queue.send
description: Process an event stream with duplicate suppression, an outbox,
  idempotent delivery and compensation.
---
function step(acc, item) -> State
  duplicate = seen(acc, item)
  if duplicate:
    return acc
  else:
    kind = read_event(item.text)
    next = transition(acc, item, kind)
    sent = dispatch(next.outbox)
    if dispatch fails: report a blocker without clearing the outbox or sending it again
    return clear_outbox(next) after dispatch succeeds
