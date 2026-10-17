# Full derivative evidence for an attention arithmetic change

The supervisor selects a potentially faster declared attention mode only after
checking the restored live model. This check extends the existing numerical
verification mechanism; it is not a new training objective or an automatic
permission to change a recipe.

1. Verify the actual production optimizer names every trainable parameter and
   is at an unarmed update boundary. Record parameter, gradient and optimizer
   tensor versions and the source RNG. Keep the model and optimizer in place.
2. Run the existing complete communication qualifier in the original mode.
   Retain its full cached-execution/transport result, including known failures.
   Reuse its exact producer/consumer case rather than constructing a second
   weaker case for the optimization.
3. Compare original versus candidate mode for the one-pass graph, then for
   explicit cached execution. Compare complete parameter and input-embedding
   derivatives through the existing streaming VJP comparator, retaining only
   one derivative set at a time. Compare hidden outputs and consumer loss too.
4. Run the existing complete cached-execution and transport checks in the
   candidate mode. Report those independently from cross-mode equivalence.
   A passing cross-mode comparison cannot erase a pre-existing cached gradient
   mismatch or establish cross-model readiness.
5. Restore arithmetic, training flags, backend state/counters and RNG on every
   exit. Verify parameters, parameter gradients and optimizer state were not
   changed. Never apply an optimizer update during qualification.

The numbered policy is natural language. Crisp code performs exact graph
reconstruction, mode selection, derivative comparisons and state preservation.
Both model families share it. Tests exercise full parameter coverage, injected
wrong derivatives, failure cleanup and actual supported CUDA modes. Hardware
throughput and an actual H100 result are required separately; local tiny-model
tests do not establish a production speedup or qualification.

The existing native SFT arithmetic-transition caller supplies the actual restored
optimizer when attention changes; unchanged attention incurs no extra check.
Channel transport's exact save/load check writes private copies of only learned
rows and the low-rank adapter. Original channel values/identities/versions and
all backbone versions remain untouched, as do optimizer moments and pre-existing
gradients. Copies must remain numerically equal; only their round-trip version
increments are permitted. Counter evidence uses the existing training-backend
qualifier and actual autocast execution, including each cross-mode VJP. Old/new
outputs use distinct slots even for an identical-mode diagnostic. All other
arithmetic stays fixed during this component comparison. No current production
runtime or recipe is changed by adding this provision.

Validation on 2026-10-17: the explicit CPU-only ledger run passed 26 tests,
including both ports, inherited optimizer state, the real SFT caller, stateful
backend ordering, exact channel ownership, injected wrong derivatives and
counter failures. Nine CUDA cases were deliberately excluded; their provision
is retained for a separately budgeted destination check. The initial name-only
CPU selector accidentally included an existing CUDA continuation test after
26 progress dots and hit its 3 GiB cgroup limit; that interrupted run is not a
suite pass. The corrected run disabled CUDA explicitly, passed in 9.41 seconds
and used 1.5 GiB peak. Quick checks passed. No production arithmetic transition
or GPU performance result is claimed by these CPU checks.
