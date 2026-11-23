---
description: Decide whether the latest revision of a value still changed what it says.
args:
  revision: Revision
returns: boolean
readout: decision
types:
  Revision: { previous: unknown, current: unknown }
---
revision.current is the next revision of revision.previous. Answer true when current says the same as previous, so
that revising again would only reword it. Answer false when current adds, removes or corrects something.
