# Account for exact captured-context render omissions

The policy remains: train only original approved actions whose complete captured input can be represented faithfully. A passing parent or materializer approval cannot supply missing provider-visible block text. Preserve the original obligation as an explicit omission and retain all independently usable actions.

1. Authenticate original source/action and existing materializer decision as before.
2. For an otherwise admitted action, execute the existing renderer's exact captured-text preparation function on the original context/tools/transport evidence. This runs before tokenization in ordinary conversion.
3. If that preparation returns an omission, reconstruct its complete typed reason/scope/block-ID/decision payload and require exact receipt equality. Do not accept arbitrary renderer exceptions, unknown reasons or receipt-only declarations.
4. Conserve approved, filtered, rendered, deduplicated and omitted original actions. Existing graph/source/profile/held checks remain independent.

The crisp mechanism reuses `captured_text_emulation_context`; it creates no new target authority, output rendering, prompt, oracle or model-facing behavior. Tokenizer-dependent render failures remain unqualified by this narrow helper. Tests use actual source-bound missing/changed block capture evidence and reject altered IDs, scope, decision, repaired capture and lost accounting.
