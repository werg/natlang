---
description: Extract and evaluate a reusable executable method from an investigation.
args:
  goal: string
  material: string
returns: Proposal
---
Look for a repeated or transferable reasoning/computation pattern in the
material. Propose a typed natlang function and any TypeScript helpers it needs
in its companion folder (helpers for `method.nl` live in `method/`).
Specify what varies, what must hold, and what result means. Save actual source
as an Edit with kind source and a .nl or .ts path, and a method description as
kind method. Include ordinary examples, counterexamples, and a held-out style
check in checks. If the available examples do not justify abstraction, return
no source edits and explain the missing evidence. Do not merely wrap a
predeclared host action. A .ts method exports `main(inputs)`. Source must run
as written, and its imports must exist or be included in edits. Record limits and dependencies.
