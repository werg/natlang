---
description: Name the product area of a support conversation and summarize what the customer needs.
args:
  messages: Line[]
  urgency: Urgency
returns: Summary
---
Summarize the open request in messages, a support conversation. urgency is how soon the customer needs an answer.
Work in these steps.

1. The open request is the customer messages that come after the last agent message; the earlier messages give
   context.
2. topic: name the product area in two to four words.
3. summary: write one sentence that says what is wrong or wanted.
4. When the customer says what they already tried, add that to the same sentence.

Return { topic, summary }.
