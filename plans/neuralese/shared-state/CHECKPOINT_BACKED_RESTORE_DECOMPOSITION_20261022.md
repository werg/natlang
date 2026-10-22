# Restore trained weights without loading discarded base weights

The current native loader reads published backbone weights, constructs the
native interface, and then overwrites the backbone with the trained checkpoint.
The same path runs during training continuations and inference startup.

1. Keep the existing exact base inventory, tokenizer and architecture checks.
2. Construct the declared backbone using its complete saved backbone state.
   Reuse Maple's existing meta construction and rotary regeneration. Use the
   installed Transformers state-dict loader for the HF backbones, with the
   pinned local config and tokenizer, without a pretrained weight source.
3. Provision the same role tokens and sparse rows, apply the saved precision
   and execution profiles, and run the existing strict complete-state restore.
4. Preserve buffer values, weight tying, tensor dtypes, numerical outputs and
   mutable model ownership. Reject incomplete or incompatible state through the
   existing strict restore; never silently fall back to unrelated base weights.
5. Verify both production families, legacy profiles and weights-only/shared
   restore paths. Measure actual restore cost before claiming a speedup.

Policy remains the owner's requirement to continue the earned model exactly.
These constructors and byte/shape checks are crisp loading mechanisms, not a
new model-selection or training policy.
