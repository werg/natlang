---
description: Draft the agent's next reply in a support conversation.
args:
  messages: Line[]
  summary: string
  missing: string[]
returns: CheckedDraft
---
Write the support agent's next reply to the customer in messages. summary is the triage of the conversation; missing
lists the details the agent still needs. Work in these steps.

1. When the conversation lets you answer, write the answer first, in one or two sentences.
2. Ask for each entry of missing, one question each, in the same order.
3. State only commitments the conversation supports: a refund the agent offered, a fix already confirmed.
4. Address the customer directly. Write two to five sentences of plain text.
