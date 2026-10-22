# Export the actual owned nested recovery

Mellum's repaired learner writes a valid published checkpoint below a nested
recovery operation inside its role-owned control directory. The existing owner
export verifier accepted only a control directory's immediate child.

1. Keep the exact byte-bound effective argv and configured output root.
2. Resolve physical paths before checking namespace ownership. For a control
   output, check the first component beneath the control root against the
   existing selected-role naming rule. Nested recovery operations inherit that
   ownership; leaf names cannot borrow another role's namespace.
3. Require the effective output to remain beneath that exact configured root.
   Preserve the existing standard, round and recovered-training forms.
4. Test both model roles, another role's root, unscoped roots, symlink escapes,
   changed argv and output escapes. Exercise the actual generated observer after
   deployment, then require independent full private readback before cleanup.

This is exact namespace/durability plumbing, shared by all model lines. It grants
no training/readiness credit and changes no model-facing behavior.
