---
description: Decide which of two equally scored schedules serves the user's request better.
readout: decision
args:
  request: string
  first: string[]
  second: string[]
returns: Choice
---
first and second are complete day plans, one line per task in time order with clock times. Both meet the request's
wishes equally well. Decide which one the user would rather have, reading request as a whole: a plan that finishes
earlier, leaves longer free blocks, or keeps related tasks close together serves most requests better, unless the
request asks for something else. Answer first when they serve it equally.
