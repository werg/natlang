# Prepare source-bound failed-history masks once per policy read

The recipe determines which exact failed-history positions have zero CE weight.
This change preserves that policy and all source, held, tokenizer and row checks.
Actual continuation stacks show dependency enumeration preparing and validating
the same complete mask manifest twice within a single call.

1. Read/authenticate the policy and prepare its masks once, retaining the exact
   dependency bindings returned by the existing mask verifier.
2. Public policy loading returns the same policy object. Dependency enumeration
   returns the same ordered policy/manifest/tokenizer references.
3. Before publishing those references, recheck each bound artifact and each
   decoder directory inventory after preparation, detecting mid-call changes.

The mechanism removes redundant full JSON/row validation. It caches no proof,
creates no new training admission and changes no weights, gradients or exposure.
It is shared by both model lines.
