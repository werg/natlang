---
description: Plan what an escalation does for a ticket that no agent answered in time.
args:
  facts: EscalationFacts
  messages: Line[]
returns: EscalationPlan
---
Plan the escalation of a ticket whose customer waited past the deadline. facts has the ticket's urgency, topic,
summary, missing details, waited_minutes, allowed_minutes and late_messages (the customer messages written after the
deadline passed); messages is the conversation. Work in these steps.

1. Write one line that states how long the customer has waited (facts.waited_minutes) and the time the ticket was
   allowed (facts.allowed_minutes).
2. notify: "on-call" when facts.urgency is "urgent"; otherwise "supervisor".
3. note: the line from step 1, then two sentences with the summary, the topic and what is still missing.
4. holding_reply: when facts.late_messages is 2 or more, write two sentences for an agent to send to the customer;
   they acknowledge the wait and give the next step. Otherwise null.

Return { notify, note, holding_reply }. The desk delivers the plan; an agent decides whether to send the holding reply.
