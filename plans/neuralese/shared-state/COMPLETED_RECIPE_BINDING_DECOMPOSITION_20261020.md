# Completed recipe artifact binding (2026-10-20)

`continueCompletedWork(earnedState, futureRecipe)` remains the existing shared
training policy. This change adds no scheduling or numerical decision. Its crisp
artifact verifier accepts either existing recipe-reference representation:
path/SHA-256, or path/SHA-256 plus an actual checked byte count. It verifies the
file and every declared field before comparing exact path/SHA identity. Unknown
fields and non-integer, negative or incorrect byte counts fail closed.

Extend the existing completed-supervised verifier at recipe-reference seams:
prepare, future-global declaration, old-global authority, and owned declaration.
All other declaration fields compare exactly. Keep the supplied receipt shape
for durable provenance; do not rewrite historical proofs, normalize arbitrary
subsets, decode checkpoints again, reset optimizer/RNG or award exposure. Real
serialized completed-quantum fixtures exercise both reference representations,
wrong bytes/hash/path and unchanged optimizer/control state. Frozen R4 remains
unchanged; this is a future shared source fix for both models.

## Requested early refresh records (2026-10-20)

The supervisor still decides when to refresh. The existing crisp completed
verifier authenticates the trainer's `StopSignal` receipt: a list of exact
`signal`/finite numeric `unix` records, including `SIGUSR2`, together with the
existing reached boundary, earned update count and declared quantum. This
extends the existing verifier; it neither rewrites historical receipts nor
creates completion or exposure credit. Tests exercise the actual signal
producer and reject malformed records before any publication changes.

## Scheduling artifact identity (2026-10-20)

The completed transition also accepts either checked artifact-reference form for
the declared future scheduling plan. Its shared verifier checks every supplied
byte count, path and SHA before comparing identity; the normal scheduling
producer emits path/SHA while the immutable future-plan proof can include bytes.
The exact future-global content and owned declaration still authenticate the
transition before any configuration write. For a preserved old scheduling plan,
authenticate its exact declaration from the immutable original configuration and
its SHA from the original global snapshot. The destination configured-plan path
may differ; it does not redefine the old authority. Serialized transition tests
exercise both destination layouts and both reference forms without cursor,
optimizer, exposure or checkpoint changes. Invalid optional byte counts fail.

## Non-admitting preference capacity history

Policy: preserve actual current admission and replay inputs; a diagnostic capacity
case audit reference explicitly marked `training_admission: false` grants no
data admission. Its historical `provenance.admission_audit` is retained in the authenticated
descriptor bytes but is not a current replay dependency. The crisp closure walker
recognizes only that exact top-level schema and the direct audit reference's
explicit false flag, then skips only that exact provenance field. A flag on the
document itself does not authorize skipping an unmarked audit reference. Other provenance, source, inputs, cases and
runtime references remain strict, including nested same-named fields. No general
provenance exemption, certificate rewrite or training credit is introduced.
