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
