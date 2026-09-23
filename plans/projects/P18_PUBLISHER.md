# P18 — Semantic document compiler and publisher

Status: local implementation in `applications/publisher/` and `applications/publisher/index.ts`.
[Shared capabilities](README.md).

The natlang entry point reads pinned source passages, plans an outline, composes
the document tree, checks it, and asks the host to prepare the result. The
host verifies literal citation quotes against current source spans, exact
table shapes, asset IDs and safe image URLs. It renders Markdown and HTML from
one portable tree. A versioned pair of files is published by one atomic local
symlink switch; an external release is not implied.

Integration tests execute the natlang function and verify both artifacts,
escaping, a table, and the publication pointer. Negative tests reject a
fabricated quote, stale revision, missing asset, unsafe URL and path escape.
Semantic claim entailment, visual review, printable output and remote release
are open product gates. Citation checking proves provenance, not truth.

## Natlang prerequisites

C0 supports outlines, section composition and consistency checks; C7 expresses a structured document. C1 exposes exact renderers and native assets in eval. C4 can supply evidence when publishing run reports. No HTML language primitive, asset type or generic publishing protocol is necessary.

## Programme and typed boundary

`publish.nl(source, audience, format, rules) -> Publication` calls `outline.nl`, `compose_section.nl`, `check_claims.nl` and `adapt_view.nl`. Exact helpers build tables, validate links and render the approved document structure.

Use ordinary records/lists for sections, claims, citations, tables, figures and code examples. Native images/fonts/render objects stay in the environment. The tree contains IDs/descriptions, not their bytes. Keep the initial document schema small enough for this product; do not try to define every possible document format up front.

## Crisp environment

A selected `ts` engine calls `render.html(document)`, `render.markdown(document)` and asset lookup helpers. HTML is an output representation here, not necessarily another executable engine. Add a print renderer later without changing the natlang type checker.

A shared environment may provide native rendering objects directly. Capture the document value and render settings for portable inspection; native layout internals need not be serialised. “Prepare publication” produces outputs locally; releasing them to an external destination is a separate explicit application action.

## Reduction and stream shape

Finite Map can compose independent sections; a later semantic pass checks cross-section consistency and evidence. Exact arithmetic and source tables are never recomputed from model prose. An Iterate can revise a draft under a bounded rubric.

Live reports later fold source-update events and publish revisioned drafts. A report remains tied to the evidence snapshot that generated it. New data does not silently alter the factual basis of an already approved output.

## Delivery and checks

1. Render a small frozen experiment result as HTML and Markdown from one document tree.
2. Add citations, tables and images. Gate: source links resolve, figures refer to actual assets, exact numbers agree across outputs.
3. Add semantic revision and visual review. Gate: claims remain supported and the requested audience/structure is satisfied.
4. Add printable output and optional publication destination as separate host contracts.

Test missing assets, broken references, conflicting numbers, code escaping, oversized sections and changed source after drafting. Renderer success alone does not establish readability or factual correctness.

## Trace and teacher

Record source snapshot, outline/section decisions, exact table derivations, revision checks and render identity. Teacher examples separate semantic composition from exact rendering. A document can be regenerated from captured portable structure even if the original renderer's native objects were shared and unrecorded; pixel equality across different renderers is not promised. This project can publish results from the others without becoming a runtime dependency for them.
