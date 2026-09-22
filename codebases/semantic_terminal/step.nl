import { cancel } from "./step/cancel";
import { explain } from "./step/explain";
import { ignore } from "./step/ignore";
import { interpret } from "./step/interpret";
import { launch } from "./step/launch";
import { recipes } from "./step/recipes";
import { recover } from "./step/recover";
import { settle } from "./step/settle";
---
description: Reduce one user or job event in a semantic terminal session. For a request about this workspace, inspect the relevant leaf under args/files (for example args/files/README.md/text) before selecting a recipe; pass the lazy dictionary only to children that need it.
args:
  acc: Session
  item: Event
  files?: Dict<File>
returns: Session
---
function step(acc, item, files) -> Session
  if item.kind is "request":
    catalog = recipes()
    recipe = interpret(item.text, catalog, files)
    return launch(acc, item, recipe)
  if item.kind is "complete":
    message = explain(item)
    return settle(acc, item, message)
  if item.kind is "cancel":
    return cancel(acc, item)
  if item.kind is "recover":
    return recover(acc, item)
  return ignore(acc, item)
