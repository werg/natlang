/*---
description: Put an entry into the calendar.
args:
  date: Text
  email: Text
returns: Bool
effects: [calendar.add]
---*/
fx.calendar.add({ date: args.date, note: args.email.slice(0, 80) })
return true
