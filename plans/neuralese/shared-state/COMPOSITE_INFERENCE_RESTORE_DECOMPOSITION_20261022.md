# Restore read-only native composite chains

Policy: an immutable parameter patch over a pruned inference parent is usable for inference when the caller explicitly requests it. A parameter patch cannot recover missing training history or grant optimizer authority. Preserve exact parent/patch identities and the existing default complete-state restore requirement.

1. Forward the public native loader's explicit allow_weights_only option into the composite loader and on to the one actual root restore. Keep the default false.
2. Apply the existing exact oldest-to-newest patch verification without relaxing parameter, model configuration, parent, tokenizer or artifact bindings.
3. If any restored parent is inference-only or optimizer-pruned, propagate both read-only indicators through every composite state. Existing serving optimizer rejection then applies to the final actor, even when a child carries training metadata or optimizer fragments.
4. Test two actual interface-patch generations on both architectures through the public checkpoint and serving APIs; compare all parameters and graph logits, assert exact composite identity and reject both default complete-state restore and optimizer actions. Full-state composites retain their existing behavior.

The mechanism extends the shared loader and existing serving guard. It adds no model-facing prompt or marker and no model-specific fork.
