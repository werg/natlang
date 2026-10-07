---
description: Draft the agent's next reply in a support conversation.
args:
  messages: { from: "customer" | "agent", text: string }[]
  summary: string
returns: string
---
Write the support agent's next reply to the customer in messages; summary is the
triage of the conversation. Answer what the conversation lets you answer, ask for
exactly the details still needed to help, and promise nothing the conversation
does not support. Two to five sentences of plain text, addressed to the customer.
