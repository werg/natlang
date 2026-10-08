---
description: Conversation.abort (spec §2.2, §5.4; R7). Decides which live tasks to abort-mark, which conversations' queued inputs to withdraw, and which tasks to wait for.
args:
  facts: AbortConversationFacts
returns: AbortConversationDecision
---
Decide the abort of conversation facts.conversation, applying THE OWNERSHIP TREE rules of the facts types. Crossing
background owners is allowed exactly when facts.background is true.

1. A task t of facts.tasks is reached when facts.background is true or t.background is false, and t.above is IN SCOPE
   of facts.conversation (crossing background owners when facts.background is true). A walk that ends at "unknown"
   is not reached.
2. mark: every reached task whose abortRequested is false.
3. withdraw: every entry q of facts.queued whose q.above is IN SCOPE of facts.conversation under the same crossing
   rule. The addressed conversation itself is always in scope, since its walk starts with it.
4. waitFor: every reached task when facts.background is true; otherwise none.

Return { mark, withdraw, waitFor }.
