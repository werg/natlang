# D31 static reference QA geometry closure

## Purpose

Make the already registered and BOTH-staged D31 static-reference QA graphs usable by the existing ordinary whole-delta geometry preparation path. This establishes exact source/row custody and actual tokenizer geometry only. It does not assert execution success, tool correctness, communication training, RL eligibility, model quality, or exposure.

## Components

1. The caller supplies the staged corpus snapshot, per-model native graph source manifest, conversion receipt, D31 admission review, render manifest and render-policy binding as exact path/SHA/byte references.
2. A static-reference-QA part verifier checks the declared corpus ID and every manifest member, validates the source and render provenance chain, and checks that the model-specific conversion receipt accounts for the exact complete train graph leaf. The admission review must state zero omissions/protected hits, whole-trajectory-only scope, zero exposure, and the explicit static-reference quality boundary.
3. The verifier loads the graph source with the current model tokenizer and role IDs, then derives actual per-row forward-token and KV-slot geometry using `NativeGraphSource`. It sums the converter's explicit per-row assistant-target and runtime-opening supervision counts; positive lower-weight context positions are not mistaken for assistant targets. It rejects imports, external state, held/test rows, duplicates, and mismatched IDs or source bindings.
4. The existing `prepare_whole_delta_geometry` compositor places the verified part in its ordered source stream, combines maxima and row indexes, and emits the ordinary whole-delta evidence consumed by the unchanged resident preparation API.

## Crisp code boundary

The staged JSON/JSONL/parquet/zstd artifacts and atomic immutable publication require exact code. Source-class policy remains in the bound admission review; the verifier only enforces its declared contract. Reuse the existing shared geometry compositor and `NativeGraphSource`; add one named source-part contract rather than teaching QA-specific behavior to the general corpus loader.

## Verification

The part tests cover a positive real-shaped contract and mutations to held
closure, rendered source split, graph source-row lineage after all enclosing
hashes are rebound, raw QA input-manifest membership, and target-token counts
with row count held fixed. The verifier also checks the current tokenizer and
role IDs, exact corpus members, source groups, one-node/self-contained graph
shape, and derives both geometry maxima from the actual parsed rows. Existing
compositor checks continue to validate full-source ordering, capacity evidence,
and current consumer indexes. The source registry, corpus manifest and source
bytes remain untouched, and no exposure is created by this operation.
