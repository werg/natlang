# Prefix adjoint requalification

The supervisor keeps Mellum prefix reuse disabled until the unchanged strict
forward and gradient gates pass. Qualification compares complete original
examples, including each example's target weights and mean loss. It does not
infer correctness from matching losses or from frozen/zero gradients.

1. Resolve the actual saved arithmetic and expert input-adjoint recipe.
2. Compare disjoint and shared layouts using the same complete examples.
3. Inspect nonzero backbone, norm and sparse-row derivatives against exact
   references before changing an arithmetic profile.
4. Independently review any repair before production adoption.

The numbered policy remains natural language. Tensor comparisons and the test
fixture's explicit trainable-parameter inventory are crisp exact verifiers.
The existing tiny Mellum loader freezes its base parameters. The ordinary
prefix gradient regression must explicitly enable those parameters and assert
that at least one backbone derivative is nonzero; otherwise its ordinary
examples contain no learned marker inputs and can compare only zero gradients.
The separate optimizer-update regression already enables parameters through the
real optimizer constructor and remains unchanged.

A read-only CPU diagnostic on the real tiny Mellum port reproduced bitwise
losses with a largest named-parameter relative gradient difference of 0.0224327
in BF16, with and without the explicit FP32 norm-gain profile. The FP32 case's
largest relative difference was 6.46e-7. This is localization evidence, not
production qualification, an FP8 measurement, or permission to widen gates.
Artifact: `/home/werg/data/dgxops/mellum-prefix-current-small-fullgrad-20261017.json`.
