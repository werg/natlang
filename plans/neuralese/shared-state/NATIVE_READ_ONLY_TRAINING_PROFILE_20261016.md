# Actual-parent memory and throughput comparison

The operating decision remains with the supervisor: compare a useful complete
pending batch at a normal checkpoint handoff, then resume useful production
training promptly. A lower memory footprint or faster diagnostic is not task
qualification and never changes source exposure credits.

The existing `eval.native_train_benchmark` now supports `--read-only`. Crisp
mechanism is appropriate here because it times and verifies the existing shared
training computation; it does not select training policy or create new targets.
The decomposition is: load the exact native parent through the shared loader;
configure the normal full-backbone/interface parameter policy without optimizer
hooks; restore the saved RNG; run `packed_graph_loss` and backward on the same
whole batch for checkpointed/chunk1024 and direct/chunk2048; compare all named
gradients and loss; report warm throughput and peak memory; restore RNG and
clear diagnostic gradients. No optimizer step, parameter perturbation, token
truncation or source-credit change occurs.

Example for a focused LFM check on the H100 at its next useful restore:

```sh
python -m natlang_neuralese.eval.native_train_benchmark \
  --checkpoint /path/to/exact-pinned-native-parent.pt \
  --graphs /path/to/exact-foundation-or-active-source-manifest.json \
  --start-index INDEX_OF_COMPLETE_REPRESENTATIVE_PENDING_GRAPH \
  --cases tokens:65536 --read-only --repeats 3 --out /path/to/report.json
```

This mode preserves the checkpoint's linear precision rather than replacing it
with the diagnostic CLI default. It currently requires FP8 off; Mellum's actual
FP8 backend comparison must use its existing restored-parent backend verifier.
Loss tolerance remains 1e-4 and full named gradient absolute tolerance 1e-5.
Mismatch or OOM is recorded and returns nonzero; it does not authorize adoption.
The first iteration reports compilation/startup separately from subsequent warm
iterations. Forward/supervised token counts and graph IDs bind the geometry.
Existing active readout chunk size is already 1024 on both production lines.

CPU fixtures use real tiny LFM and Mellum backbones, the actual production
parameter policy and whole producer graphs. They verify unchanged parameters,
restored RNG/training/checkpoint state and complete named gradient membership.
The BF16 fixture compares checkpointing at the same readout chunk; its chunk
arithmetic is not assumed equivalent to LFM FP32. Target full-size performance
and numerical qualification remains to be measured at the useful H100 handoff.
