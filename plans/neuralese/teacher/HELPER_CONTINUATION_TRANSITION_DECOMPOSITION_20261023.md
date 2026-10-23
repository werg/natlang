# Teacher helper continuation transition decomposition

## Goal and policy

Allow source-admission operators to advance from the exact reviewed legacy helper
to a reviewed successor while preserving the execution identity of already
converted or validated tranches. A historical receipt remains historical; it
must never be rewritten to claim the successor helper ran. New tranches use the
successor only after its complete operator closure and provenance are pinned.

Policy is expressed as a typed natural-language decision:

1. Read the helper continuation binding from the immutable conversion receipt.
2. If it equals the exact legacy binding, verify it against the preserved legacy
   observation and immutable helper authority. Treat it as historical execution
   provenance, and continue all independent source, held, protection, conversion,
   and custody checks.
3. If it equals the one explicitly declared successor binding, verify the
   successor transition receipt against the old observation, old immutable
   authority, exact new helper source, complete implementation inventory, and
   unchanged source-closure accounting. Continue the same independent checks.
4. Reject every other binding, incomplete transition, changed old receipt, or
   mismatch between the converter's receipt and validator's selected helper.
5. Never infer training admission, exposure, source quality, or held-data safety
   from a helper-continuation proof. Those remain separately verified by the
   existing source and conversion pipeline.

The policy has one shared interface and a crisp implementation because helper
selection and source-file custody are exact security boundaries. The operator
declaration pins the transition receipt; the verifier checks its content against
fixed old and new authorities rather than trusting a self-declared `passed`
field. A small crisp verifier performs byte/hash checks and JSON schema checks.
The converter and validator remain the real natlang-data pipeline; this change
does not alter how rows, actions, masking, held sources, or admission decisions
are interpreted.

## Components

| Component | Representation | Reason |
| --- | --- | --- |
| Historical-versus-successor helper policy | Shared natural-language policy contract | It decides which reviewed helper lineage a receipt belongs to. |
| Transition receipt verification | Shared crisp Python verifier | Exact paths, hashes, byte counts, JSON fields, and authority comparisons must be deterministic. |
| Operator declarations | Immutable JSON bindings | They select the one allowed successor proof and closure. |
| Converter/validator closure loading | Existing frozen sibling-loader mechanism | The API actually executed must match the reviewed source binding. |
| Prior receipts and conversion outputs | Immutable historical artifacts | They retain the helper actually used and remain valid only under that authority. |
| Source/held/protection and admission checks | Existing verifier pipeline | Helper transition is not permission to relax or replace these checks. |

## Compatibility proof contract

The successor proof binds the exact legacy observation and immutable legacy
authority, the exact successor helper path/hash/byte count and commit, the
complete source-closure implementation inventory, the unchanged closure
accounting receipt, and the independently reviewed operator closure. Each
updated converter and validator closure pins that proof's own exact file binding.
The verifier accepts only the exact legacy binding or exact successor binding;
it does not accept a generic `compatible` flag or an arbitrary proof path.

Existing conversion receipts and artifact bytes remain untouched. An updated
validator can verify a legacy receipt as historical by checking its exact old
continuation binding against the preserved old authority. A new converter emits
only the successor binding after loading and checking the successor sibling API.
An updated validator then requires receipt and execution metadata to agree and
checks the same successor proof. Any operator, runtime, held/source, masking,
or output identity remains under its pre-existing verifier.

## Deployment boundary

The implementation produces a reviewable patch and deterministic staging
manifest for the exact active converter, validator, and drainer declarations.
It does not edit live operators, rewrite in-flight evidence, activate a new
service, or merge the canonical helper. Cutover is complete only after the
operator independently stages the full closure, verifies post-loader identities,
and declares the new continuation for future work while retaining the old
authority for old outputs.
