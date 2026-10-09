---
description: Read a support desk's written service policy and say how long a ticket may wait for its first answer.
args:
  urgency: Urgency
  topic: string
  policy: string
returns: CheckedMinutes
---
Say how many minutes a ticket with this urgency and topic may wait for an agent's first answer, under policy, the
desk's written service policy. Work in these steps.

1. Find the rule in policy that names urgency. Find any rule in policy that names topic.
2. When both rules name a time, use the shorter time.
3. When one rule names a time, use that time.
4. When no rule names a time, use the default time that policy states.
5. Write the time as a whole number of minutes (one hour is 60 minutes, one day is 1440 minutes) and return it.
