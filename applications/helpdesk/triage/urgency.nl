---
description: Decide how urgently a support conversation needs an agent's first answer.
args:
  messages: Line[]
returns: Urgency
---
Decide the urgency of the open request in messages, a support conversation. Work in these steps.

1. The open request is the customer messages that come after the last agent message. When no agent has written yet, it
   is every customer message.
2. List what the customer says has happened: what stopped working, what was lost, what was reported.
3. Return "urgent" when the list contains one of these events: the product cannot be used at all; money or data is
   being lost; a security problem is reported.
4. Return "low" when the open request is a question, a suggestion or praise, and waiting a day changes nothing for the
   customer.
5. Return "normal" for every other open request.

Judge by the events the customer reports.
