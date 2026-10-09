---
description: Order the support desk's open tickets for the agents' queue under the desk's ranking rules.
args:
  tickets: RankedTicket[]
  policy: string
returns: string[]
---
Order the ids of tickets for the agents' queue, under policy, the desk's ranking rules. Work in these steps.

1. Apply the rules of policy to the tickets, reading each ticket's urgency, topic, summary, escalated and due.
2. A ticket with a smaller due time (milliseconds) is due sooner; a ticket with due null is due last.
3. Where policy does not decide between two tickets, put the one due sooner first, then the more urgent one, then the
   smaller id.
4. Return the ids of every ticket in tickets, each id once, in the queue order.
