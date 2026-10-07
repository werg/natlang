---
description: Judge how urgently a support conversation needs an agent's first answer.
args:
  messages: { from: "customer" | "agent", text: string }[]
returns: { urgency: "low" | "normal" | "urgent", topic: string, summary: string }
---
Read the support conversation in messages and judge how soon the customer needs an
agent's answer.

- "urgent": the customer cannot use the product at all, is losing money or data,
  or reports a security problem.
- "low": a question, suggestion or praise where waiting a day does no harm.
- "normal": everything else.

Judge by what the customer says has happened, not by how strongly they say it.
topic names the product area in two to four words. summary is one sentence an
agent can act on: what is wrong or wanted, and anything the customer already tried.
