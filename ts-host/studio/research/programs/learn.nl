---
description: Extract and evaluate a reusable executable method from an investigation.
args:
  goal: string
  material: string
returns: Proposal
---
Look for a repeated or transferable reasoning/computation pattern in the
material. Propose a typed natlang function and any crisp helpers it needs.
Specify what varies, what must hold, and what result means. Save actual source
as an Edit with kind source and a .nl or .ts path, and a method description as
kind method. Include ordinary examples, counterexamples, and a held-out style
check in checks. If the available examples do not justify abstraction, return
no source edits and explain the missing evidence. Do not merely wrap a
predeclared host action. Source must be executable within the selected engine
and imports must exist or be included in edits. Record limits and dependencies.
