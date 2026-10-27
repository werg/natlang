---
description: Choose the operation for each focused binding.
args:
  focus: string[]
  view: EvidenceView
  history: AdaptationState["history"]
returns: RevisionPlan
---
Choose how to revise each binding in focus. The operations are: "reencode" (rewrite its native instruction state in
the current context), "replace-skill" (rewrite its skill states), "weights" (change the LoRA mixtures on its path) and
"keep".

1. For each binding in focus, look at its failed attempts in view.items.
2. When its attempts fail with errorKind "wrong-value" or "missing-effect", choose "weights" for that binding: the
   behavior itself must change.
3. When they fail with errorKind "type" or "tool", choose "reencode": the instruction needs a clearer native state.
4. Otherwise choose "reencode".
5. Every binding outside focus keeps its current entry: choose "keep".
6. Return { operations, reason }, where reason names the bindings and the error kinds that decided them in one
   sentence.
