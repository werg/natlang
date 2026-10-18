# Scoped module forward restoration

1. The existing cotangent scope owns temporary module routing; it remains the single implementation of this policy.
2. Crisp attribute custody records whether each temporary forward existed on the instance and its exact prior value. Restore that value or remove the temporary attribute on every scope exit. This preserves Python descriptor lookup and repeated-scope behavior.
3. Linear carrier construction drops a copied instance forward so the carrier class implementation handles its temporary projection. Only an exact bound nn.Linear.forward alias is accepted; arbitrary instance overrides fail before mutation so their semantics are never silently replaced. The original module and its alias remain intact outside the scope.
4. Verify two consecutive actual VJPs, outputs, input gradients, original parameter objects, and exact forward attribute presence/value for Linear, Embedding, Conv1d and RMSNorm, with and without instance forward aliases. Existing functional and compiler checks remain applicable.
