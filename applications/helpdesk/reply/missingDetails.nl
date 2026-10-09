---
description: List the details an agent still needs from the customer to resolve a request.
args:
  messages: Line[]
  summary: string
returns: string[]
---
List what the agent needs from the customer, given messages (a support conversation) and summary (its triage).
Work in these steps.

1. Read summary and the open request: the customer messages after the last agent message.
2. List what an agent needs in order to resolve the request: for example the product version, the account, the steps
   already taken, the error text.
3. Remove every entry that the conversation already answers.
4. Return the remaining entries, one short phrase each, most important first. An empty list means the agent can answer
   now.
