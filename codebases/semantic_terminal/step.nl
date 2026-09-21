---
description: Reduce one user or job event in a semantic terminal session.
args:
  acc: Session
  item: Event
returns: Session
---
function step(acc, item) -> Session
  if item.kind is "request":
    catalog = recipes()
    recipe = interpret(item.text, catalog)
    return launch(acc, item, recipe)
  if item.kind is "complete":
    message = explain(item)
    return settle(acc, item, message)
  if item.kind is "cancel":
    return cancel(acc, item)
  if item.kind is "recover":
    return recover(acc, item)
  return ignore(acc, item)
