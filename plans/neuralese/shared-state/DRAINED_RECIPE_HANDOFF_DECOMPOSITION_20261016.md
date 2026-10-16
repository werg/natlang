# Drained recipe handoff into raw training

Policy: after its actual optimizer owner drains, continue the same student with
reviewed runtime and source scheduling improvements. Preserve the original
handoff, resident history, checkpoint, optimizer/RNG and exposure. Never rewrite
a historical resident commit to make it claim the new policy.

1. Authenticate the original owner drain and exact saved resident generation.
2. Require the existing retirement record and absence of a live/current owner.
3. Resolve the new recipe and run the existing lineage_handoff verifier against
   the final adopted old plan, including exact priority-only continuation checks.
4. Persist the handoff using that same implementation, retain its old plan and
   frozen runtime, and bind the new plan, recipe, source inputs and runtime in an
   immutable receipt. No model update occurs.
5. Reverify the receipt before bootstrap, choose the new recipe explicitly, and
   retain the old history as provenance. The new recipe becomes the subsequent
   resident release origin; the old checkpoint continues unchanged.

These are exact durability and authentication helpers. The decision to drain or
change the declared recipe remains the supervisor's natural-language policy.

## Initial-only owners after a completed handoff

A restarted trainer has a real initial owner and installed recipe plan before
any new resident append exists. A subsequent certificate-only recipe change
must use this actual authority, without inventing a resident generation or
retirement. At safe drain snapshot the installed plan; verify its digest against
the actual initial receipt, exact saved recipe binding, stage, and train/held
source receipts. Reuse the existing lineage verifier/atomic plan commit and
independently recompute it on reload. Version2 recipe proofs distinguish this
case from generation retirement; historical version1 proofs retain their scope.
Completed model/optimizer/RNG/exposure and process-tree checks remain mandatory.
