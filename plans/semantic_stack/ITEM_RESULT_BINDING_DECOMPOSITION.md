# Actual child-result binding for one-item natlang frontiers

## Native program and boundary

The source program owns every semantic decision. A named one-item reducer takes a typed frontier, selects one original item, awaits its scoped natural-language judgment, appends that actual result, and returns a smaller frontier. Its parent owns `iterateOn`. The selected item, judgment, and later audit remain natural-language stages with numbered policy; an exact checker does not choose an edge, scope, effect, or answer.

The checker is crisp because it compares runtime-observed typed values and invocation lineage. It answers only whether the result slot came from the actual scoped child of this reducer on this original item, with earlier completed slots unchanged. A semantically plausible object manufactured in the parent does not satisfy that provenance invariant.

## Existing mechanism to extend

- `exactHostTraceCapture` and `hostCaptures` already snapshot portable outputs with call IDs and parent IDs. The teacher collector currently enables all outputs but no inputs (`definitionSources:[]`, `inputArguments:[]`); the completed `invocation_ledger` therefore cannot prove which item a scoped child received.
- Add a narrow opt-in capture for declared reducer `state` and its inline scoped child input, bounded by the existing byte limit and persisted before CallStore eviction. Persist complete/incomplete status; an absent, truncated or nonportable input yields `unverified_hold`, not a model-negative judgment. Do not synthesize a model action or display a new tool name.
- Extend the existing collector ledger with those host-only input captures. Use the runtime's parent/child call IDs and source/inline-site identity. Do not match results merely because their text is equal.

## Exact qualification

Read each declared site in `ASTRA_ECOLOGY_ITEM_RESULT_V18_PROPOSAL.json`. Resolve its RFC6901 pointers against the named reducer's captured input, scoped child's captured input and output, and named reducer's actual output. Check one original item consumed, one completed identity/result appended, unchanged prior prefix, stable declared fields, and exact typed semantic fields before any explicitly declared parent witness projection. Apply the declared item identity comparisons. A terminal no-item branch is checked separately and cannot earn semantic-child credit.

Emit `pass`, `fail`, or `unverified_hold` as a diagnostic qualification separate from source replay, oracle equality, and training admission. A failed provenance check holds that trajectory for repair; it does not make the underlying source task or optimizer approach invalid. Keep the exact observed inputs/outputs and call IDs in the receipt so a reviewer can inspect the boundary.

## Controls and rollout

Use a canonical one-item call with a real scoped child, plus negative mutations: parent-made answer without child, wrong child item, stale copied child output, altered semantic field, dropped prior result, repeated original item, missing input capture, and unrelated sibling child. Verify existing collector output and authored source bytes remain unchanged when this opt-in is absent. No model-facing feedback or automatic recovery is added in this unit; if later introduced, update both natlang skills and measure about 48 live samples before adoption. The v18 source presentation has its own live measurement debt.
