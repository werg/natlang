# Held static IR from typed decision labels

`teacher-decision-labels.mjs` converts existing decision-label JSONL into ordinary
`natlang.program/2` curriculum records. It authors the controller and reference
scaffold; it does not claim that the provider wrote either. A provider's
probabilities are retained as a raw receipt and reduced to the selected option
(or, for `noul`, the Boolean `p >= 0.5`) for each inline child result. The
runtime-composed instruction is kind-specific: binary cases ask for `true` when
yes is at least as likely as no and otherwise `false`, with no numeric
probability; choice and ordinal cases ask for exactly one declared option or
level, with no probability distribution. The source question itself remains
verbatim in the runtime-composed instruction.

Example:

```sh
node scripts/static-adapters/teacher-decision-labels.mjs \
  --cases ../runs/free-provider-generation-20261009-v1/throughput-mixed-decisions-source-v1/cases.jsonl \
  --labels ../runs/free-provider-generation-20261009-v1/provider-google-gemini-3.5-flash-lite-mixed-decisions-v1/labels.jsonl \
  --out ../runs/free-provider-generation-20261009-v1/held-static-decision-labels-review-v1 \
  --batch-size 8 --replay-limit 3
```

The source directory must contain `source-manifest.json`; each label file must
have its `<labels>.manifest.json` sidecar, and both must pin the exact case-file
SHA-256. It accepts the current `natlang.decision-labels/1` single-model
manifest and `natlang.gemini-decision-pool/1` per-row pool manifest. Each row's
provider model, endpoint, and cases SHA must bind to that manifest; the pool
row's teacher is retained and checked against its actual listed model. The
output directory must be empty. Output files are created
exclusively, so a review snapshot is not silently overwritten.

Only source `train` rows are candidates. A candidate is retained only when the
answer payload has exactly the expected probability keys, numeric finite values
in `[0,1]`, and a total within `1e-4` of one; `scoreGraded` reports every gate
true; and the deterministic highest-probability option/level (first source
option wins a tie) or binary threshold agrees with the source annotation. Numeric
ordinal annotations are zero-based positions, not literal level names. The
adapter uses the scorer's shared `ordinalDistribution` for the annotation too:
integer positions map to their named level; fractional positions split mass
between neighbours and select the nearest level (the earlier level on an exact
tie). This is an explicit crisp reduction, not a modification of the annotation
or teacher probabilities. The
adapter never normalizes probabilities or repairs labels. Rejections record
missing labels, typed errors, strict-schema failures, gate failures, and
source-label disagreement.

Compatible rows are grouped by source, family, kind, exact options/levels,
criteria, license, role, and source group. Source criteria are kept verbatim
when present; a label-name criteria map is generated only if the source has no
criteria. Each authored directory reducer binds case ID, state, question,
criteria, options, and levels as separate values passed to a typed inline `nl`
call. The item question is also interpolated into the NL instruction at runtime;
that runtime-composed instruction is distinct from the separate scope arguments
that carry question, state, criteria, options, and levels. No serialized prompt
variable combines those fields. The controller aggregates child results in
code. The model-facing folder contains
all reviewed source fields except the source answer annotation. Unknown source
fields are rejected from prompt exposure and listed in `rejections.jsonl`.
Probability receipts and expected child outputs remain oracle/provenance data.

The replay result includes a capture-visibility record derived from actual
reference turns; all captured values must appear in the child context for the
sample replay to report visibility success. A short eval logs those captures
before the scripted child answer, rather than returning an unrelated object as
the child's typed answer.

`cases.jsonl`, manifests, replay rows, and native rows all carry an explicit
`training_admission: false` or equivalent held disposition. The optional
reference replay is a scripted static reference with zero provider calls. Its
materialized rows are marked `authored-static-reference-pending-review` and
`authored-static-reference-not-teacher-trace`. Passing a runtime replay is not
semantic validation, teacher-trace evidence, or training approval. A reviewer
must make those decisions separately.
