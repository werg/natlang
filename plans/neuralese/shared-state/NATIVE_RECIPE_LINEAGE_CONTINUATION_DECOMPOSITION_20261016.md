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

## Implemented raw/continuation repair

The release requirement's `recipe_sha256` intentionally remains the CURRENT
recipe policy identity. It is not replaced with the preserved lineage. A new
optional `recipe_lineage_sha256` checks lineage separately, so an old-policy
readiness receipt cannot qualify a new recipe merely because both share history.
Raw configuration obtains lineage from its actual immutable installed/current
plan and verifies saved checkpoint binding consistency. Initial-only owners also
supply their actual installed recipe and train/held sources; no resident current
or adoption generation is manufactured. Existing raw configs acquire missing
lineage metadata only after sealed operations finish, retaining task counters,
current checkpoint, optimizer and source credits.

Both RL and seed trainer parameters carry the current recipe binding plus the
preserved lineage. Subsequent mandatory CE/preference quanta preserve that lineage
while rebinding current policy. Direct native trainer continuation inherits a
saved binding when no override is supplied, and rejects an explicit lineage
relabel. Resident source adoption updates both trainer bindings under the same
lineage. Normal recipe restart accepts an already committed exact handoff only
when the resolved policy matches and the last handoff records the requested
current recipe identity; every other plan field and frozen code inventory still
must match. Unapproved policy changes continue to require explicit handoff.

Qualification: `native-lineage-propagation-final-cpu-20261016`70 PASS/1 CUDA skip,
14.85s/794MiB; `native-lineage-initial-config-final-cpu-20261016`25 PASS5.83s,
including initial-only binding/source selection and current-policy readiness
negative. `native-lineage-fourphase-cuda-20261016`2 PASS29.67s/3GiB: actual LFM
and executed-FP8 Mellum SFT→RL→SFT→RL preserve optimizer/RNG, global steps1..4,
source credits1/1/2/2 and a recipe lineage distinct from current file SHA. An
explicit relabel attempt rejects without modifying the parent checkpoint.

The adjacent fresh shared-semantic bootstrap/export must also carry per-model
recipe provenance. That metadata path is separately identified for closure;
these raw/continuation checks do not claim it already inherits lineage from a
semantic export that never recorded it. No model arithmetic, source-policy,
reward/whole-episode eligibility or model-facing prompt changes occur here.
