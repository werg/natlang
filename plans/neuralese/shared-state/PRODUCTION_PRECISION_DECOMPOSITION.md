# Native production throughput and precision

Policy: use the fastest measured arithmetic that preserves useful training,
valid consumer-to-producer gradients and working cached/shared-state inference.
Keep strict arithmetic as an explicit diagnostic reference. Source precision and
source identity remain exact; runtime arithmetic may adapt without throwing away
weights, optimizer moments, RNG or recorded data exposure. Production changes
follow measured kernel/whole-graph evidence and a declared checkpoint/recipe
transition. They do not silently redefine strict-reference tolerances.

Exact mechanisms: profile the shared trainer's graph forward, loss/readout,
backward and optimizer separately; identify the actual expensive operators;
compare declared tensor-core and mixed-forward modes on both backbone ports
against strict CE, gradients and cached/flat behavior. Share precision selection
and serialization across native factories, SFT and inference. Numerical kernels,
state identities and resume transition verifiers are crisp mechanisms; the
throughput-first adaptation policy is stated above.

The supervisor's natlang policy is:

1. Read both models' actual update throughput, allocated/reserved memory, loss,
   task results and pending complete source exposure.
2. Identify the expensive operators and measure a relevant whole-graph update.
   Preserve complete trajectories; larger packing changes scheduling rather than
   dropping rows or truncating context.
3. Select the fastest supported setting with valid communication gradients,
   cached execution, held separation and actual task behavior. A small base
   distribution change is telemetry. Investigate failed runtime parity as a bug.
4. Change the shared recipe at an update boundary, preserving weights, optimizer,
   RNG and row credits. Prewarm the exact code/data closure while useful training
   continues. Measure the resumed production updates and keep adapting.

`native_precision.py` provides exact setting/identity mechanisms, not a policy
that silently chooses a mode. Stock torch math has one common IEEE process
policy. Each model independently selects its native projection/readout kernels,
so loading the second participant cannot override the first participant's mode.
The low-rank embedding adapter, raw KV/hybrid state and gradient junctions remain
FP32; stock FFNs, routers and optimizer retain their existing shared mechanisms.
No module or parameter is replaced, and state-dict/optimizer parameter names stay
stable. Activation-checkpoint recomputation sees persistent per-module settings.

`bf16x6` decomposes each FP32 operand into three BF16 pieces and retains six
products (high-high, both high-middle, both high-low and middle-middle), with
FP32 accumulation/output and FP32 master parameters/gradients. This is declared
approximate FP32 arithmetic, not an assertion of universal IEEE equivalence.
Qualification still compares actual flat/cached forward, consumer CE, producer
gradients and incremental splicing with the unchanged numerical tolerances.
The strict `ieee` mode and historical strict checkpoint identities remain usable.
Coarse TF32/BF16 remain available to the diagnostic kernel harness; they are not
production recipe options because the real full model exposed parity failures.
An explicit finite-convolution row reduction in the new modes removes the
cuDNN-versus-graph reduction discrepancy rather than hiding it with tolerances.

The SFT CLI/recipe setting is `native_linear_precision` (`ieee` or `bf16x6`).
Checkpoint configuration and channel metadata bind the selected arithmetic.
Native block embedding-path identity includes nondefault arithmetic, preventing
state produced under another mode from being silently accepted. Restoration
retains the saved mode; an explicitly requested update-boundary transition
records its source dtype, old/new mode and global update in persistent training
history. Model inventory, optimizer, RNG and row-exposure verifiers still apply.
A mode transition does not create a fresh lineage or reset data progress.

`native_training_update` is the existing complete SFT update extracted into one
shared primitive. The production loop and `eval.native_train_benchmark` both
use it: identical objective, layer checkpointing, in-backward MuonSR, FP8
backend, finite-gradient checks and ordinary optimizer step. The diagnostic
wrapper reports cold and warm whole-graph update times, actual token/graph
volumes and per-case allocated-memory peaks. Its temporary updates never count
as production exposure or publish trained checkpoints. `eval.native_throughput`
separately measures full-layer backward and cached/reference numerical behavior
without optimizer steps. Neither wrapper is a second trainer.

Measured on DGX GB10, 2026-10-15:

- The original 6,958-token genuine communication update profile used the real
  shared optimizer: warm forward 0.874s, backward+optimizer 2.716s. Stock matrix
  multiplies and custom IEEE GEMMs each accounted for approximately 25% of GPU
  time; attention backward accounted for approximately 14%.
- Native-only `bf16x6`, complete QAD FP32 backbone backward, chunk 1,024:
  3.281s IEEE versus 3.064s (about 7% gain). Complete-graph CE changed by
  -3.34e-6; producer-adapter gradient relative L2 changed by 1.07e-4. Cached
  forward/CE/gradient/incremental errors remained below 1.2e-6.
- Full Mellum BF16 backbone: complete-graph CE and producer gradients were
  bitwise identical between modes; cached/reference/incremental errors were
  zero. BF16 native projections already use tensor cores, so no arithmetic
  throughput improvement is claimed for Mellum from this setting.
- IEEE chunk 128 versus 1,024, all 1,149,578,240 bytes of trainable gradients:
  maximum difference 7.45e-9, relative L2 5.32e-11, producer adapter bitwise,
  CE difference 4.77e-7. Full-backward timing improved only about 2.6% on that
  graph. Earlier adapter-only timings overstated the full-training benefit and
  must not be quoted as production throughput.
- TF32x3 was approximately 33 times slower than IEEE in the GB10 microkernel;
  a hardware name alone is not a performance certificate. Replacing the stock
  FFNs with custom GEMMs also lost throughput and was removed.

Receipts are in `/home/werg/data/dgxops/`: `native-lfm-production-profile-20261015.json`,
`native-lfm-production-precision-full-training-20261015.json`,
`native-mellum-production-precision-20261015.json`, and
`native-lfm-full-gradient-chunks-20261015.json`. Actual packed optimizer profiles
are recorded separately in `native-*-packed-production-20261015.json`; use those
and resumed CoreWeave telemetry to choose operational packing/chunk settings.
These receipts establish numerical/runtime behavior, not task lift or production
completion. Task/held evaluation and whole-recipe exposure remain required.

Outstanding inference completeness work remains the advanced learning-service
persistence/adapters/order-two/guidance integration. This throughput change does
not claim those independently tracked gaps are complete.

Arithmetic transitions to bf16x6 check the actual restored destination weights
against the existing cached/reference forward, consumer loss, producer gradient
and incremental splice checks before the first optimizer update. The trainer
reuses its loaded model and FP8 context, preserves RNG, retains exact optimizer
and row credits, and stores the transition check with its parent weight digest.
This is a numerical transition check, not full-stage/task qualification. Unchanged
modes incur no extra reference pass. Tolerances remain explicit recipe inputs;
a failed check leaves the previous checkpoint available for IEEE continuation.

The reference probe derives producer bounds and the consumer import position
from actual tokenization, including one-token vocabularies. If its diagnostic
sentence maps entirely to special/unknown tokens, it round-trips an existing
lexical vocabulary entry; it never fabricates a payload state or relaxes a zero
producer-gradient failure. Normal production tokenizations retain their probe.
