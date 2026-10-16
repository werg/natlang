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

Shared semantic continuation carries each participant's own current recipe and
preserved lineage in its `training_options`, and the adapter export repeats that
exact metadata. The existing selected participant loader exposes these options;
the joint semantic curriculum identity never replaces a model's recipe lineage.
Actual participant load rejects manifest/checkpoint disagreement. Resume retains
metadata on unchanged participants and requires the same lineage on an explicitly
optimizer-preserving changed participant; a new current recipe remains distinct.

Fresh and reused semantic RL bootstrap verify adapter metadata against the actual
selected checkpoint payload before building both seed and RL trainer bindings.
These bindings reference the supplied current recipe under the preserved lineage.
Operator settings survive reuse; unchanged configs are not rewritten. Semantic
refresh carries exact updated receipt metadata and unchanged exported metadata
into its next participant manifest. Provenance-only historical recipe paths do
not become new runtime dependencies. A legacy checkpoint missing the metadata
cannot invent a lineage from an adapter: recovery requires its exact original
qualified readiness and checkpoint binding, otherwise it must be repaired before
production bootstrap.

The policy remains natural language in the existing recipe. These helpers are
crisp identity readers, exact mismatch verifiers and durable metadata plumbing.
No model arithmetic, source-policy, reward/whole-episode eligibility or
model-facing prompt changes occur here. Current-policy readiness checks remain
mandatory independently of preserved lineage.

Semantic qualification: `native-semantic-lineage-final-cpu-20261016`42 PASS,
16.70s/953MiB, including actual both-port shared payload/bundle reload,
checkpoint-versus-export forgery rejection, actual participant load and bootstrap
hot-config retention. `native-semantic-lineage-resume2-20261016`3 PASS3.83s:
updated checkpoint missing metadata, changed lineage and unchanged weights with
relabeled policy all reject without changing the parent. The first invocation
of those three new tests used a fixture output path occupied by its source file;
that fixture path was corrected before these terminal results.
`native-semantic-lineage-final-cuda-20261016`1 PASS19.30s/3.0GiB: real LFM and
executed-FP8 Mellum VAE/joint/distill/policy training, stop/resume and extended
policy epochs preserve exact participant recipe metadata, optimizer, RNG and
all existing exposure obligations. `native-semantic-lineage-quick-20261016`
passed22 ratchets and TypeScript checks. No full-model cloud cutover or semantic
foundation completion is claimed by these tiny actual-port checks.

Independent-review repair: original readiness policy on legacy backfill

The initial legacy metadata recovery constructed an incomplete readiness
requirement. It omitted `stage_ids` and numerical/source policy fields required
by the shared verifier; that route could not complete. The verified participant
record now carries the supervisor's original stable `readiness_requirement`:
model/current recipe identities, optional preserved lineage, required stages,
forward/gradient tolerances, and the declared foundation source/epochs/boundary.
Only the shared readiness verifier creates this record. Legacy backfill requires
that manifest-bound policy and reruns the same verifier; it cannot select a
stage, foundation cohort or tolerance from the receipt it is authenticating.
Owner-process paths are not copied into this historical policy, because original
ownership was checked before manifest publication and may legitimately move.
An old manifest without the declared recovery policy must be regenerated from
its original valid readiness requirement before recovering absent metadata.

`native-semantic-lineage-legacy2-20261016`2 PASS2.71s uses a real tiny LFM model
and actual optimizer checkpoint missing recipe metadata. Both full-stage and
communication-foundation recovery run the real readiness verifier without a
mock. Exact metadata and optimizer values survive. Missing policy, wrong stage,
changed declared stage inventory and changed bound foundation-source metadata
reject. The initial foundation test invocation supplied an empty ordinary
source to a helper that requires nonempty files; the fixture was corrected to
include an actual ordinary context row. Protocol receipt fields in this test
are synthetic evidence fixtures, not claims of production numerical/task or
source-exposure qualification. No model arithmetic is changed.
