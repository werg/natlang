---
description: Judge how well one complete schedule meets one wish of the user.
readout: decision
args:
  schedule: string[]
  preference: Preference
returns: Fit
---
schedule is a complete day plan, one line per task in time order with clock times ("draft 09:00-09:30"). preference is
one wish of the user. Decide how well the plan meets it. met: the plan does what the wish says for every task it
concerns (a wish about the whole plan concerns every task). partly: it does so for some of those tasks, or comes close
(for example 30 minutes away from the time the wish names). missed: it does the opposite or nothing the wish asks.
Judge only this wish; other wishes are judged separately.
