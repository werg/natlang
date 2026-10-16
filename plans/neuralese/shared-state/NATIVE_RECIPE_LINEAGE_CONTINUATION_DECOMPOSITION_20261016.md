# Carry current policy and preserved lineage through raw/shared continuation

The declared natural-language recipe chooses training work. These helpers make
no new scheduling decision: they authenticate and carry the exact current recipe
content/file binding separately from its preserved optimizer/run lineage.

1. An installed plan's recipe_sha256 is its preserved lineage identity after an
   explicit handoff. Native recipe bindings carry current sha256/file_sha256
   alongside lineage_sha256. Verify both roles; never infer that changed policy
   bytes created a fresh optimizer lineage.
2. Initial-only drain authority compares its saved and installed lineage fields,
   retaining exact saved recipe/stage/train/held/current-policy checks and the
   original immutable owner/checkpoint evidence.
3. Raw configuration inherits the verified new plan or actual resident plan's
   lineage. Supervised continuations bind the current policy path and this same
   lineage; source/optimizer/RNG/exposure state remains carried by shared loaders.
4. Readiness continues checking CURRENT recipe policy identity, with a separate
   lineage assertion when declared. Equal lineages cannot authorize old-policy
   readiness for a changed current recipe.
5. Normal recipe restart reuses an authenticated installed identical plan's
   preserved lineage rather than silently asking for a new lineage. Existing
   explicit handoff validation remains required for actual code/policy changes.

Crisp code is required for SHA/plan/receipt verification and exact continuation
plumbing. There is no model-facing prompt, model arithmetic or source-policy
change. Independent review precedes operational adoption.
