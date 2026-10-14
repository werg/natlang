---
description: Build actual quote, reference and illocution branches per original turn.
args:
  source: DialogueSource
returns: ScopedConversation
---
1. Keep the finite original turn array. For each turn call partitionQuotation(turn, source.rules,
   source.temporal), retaining its actual QuoteTree under that original turn identity.
2. Call resolveReferences(turn, quoteTree, source). Pass the actual attributed frames; do not create an
   easier flat transcript or let a forwarding timestamp replace a quoted original time.
3. Call classifySpeechActs(turn, quoteTree, references, source). The returned acts own speaker,
   proposition, endorsement and condition meaning; neither this parent nor a type table re-derives them.
4. Join returned act IDs and reference span IDs to actual nodes of that turn. Nonempty semantic
   contributions require exact original support. Unknown attribution stays a nullable typed contribution.
5. For each actual joined turn directly invoke a scoped NL LedgerFinding literal capturing returned
   quote attribution, reference meanings, performed/depicted act readings and original context. Judge
   whether these contributions express one coherent scope. An actor's explicitly mistaken private belief
   can be a valid perspective difference; a child assigning the reporter the quoted original's promise
   is an inconsistent output. Retain the actual witnessed finding in scopeFindings without overwriting
   the child results. A missing premise leaves pass null and its original IDs unresolved.
6. Retain ambiguity alternatives and unresolved child reasons separately from known report/quote status.
   A source-established forwarding-only act is a resolved non-endorsement, not an unresolved assertion.
7. Return all actual quote trees, reference bindings, act readings and scopeFindings. Equal quoted strings in different
   turns keep separate frames and availability paths; only their original target identity may be shared.
