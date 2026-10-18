# Convolution shared-state fan-in: localized cause and exact-parent diagnostic

The remaining gradient discrepancy in the existing fixed-FFN/fused-norm diagnostic is caused by shared convolution-state gradient accumulation. It is not evidence against the shared-state architecture. This finding does not deploy new arithmetic or qualify the current cloud runtimes.

The source is the actual LFM checkpoint 9478, SHA-256 `f07a57e3d98224360b8a509f8f410be3de2fd37830e176c989164d2e83ebcb89`, at `/mnt/external/natlang-development-data/lfm9478-paused-quality-recovery-20261017/checkpoint.pt`. The existing complete communication reference has 32 physical tokens and 151 named/input derivative comparisons. Backbone projection precision remains `bf16x6`; no optimizer update is performed. These diagnostics construct a fresh comparison optimizer; they do not qualify restoration of the saved optimizer history.

The first observational trace found identical forward values throughout all 16 layers. At the final convolution layer, output, `c`, and history-tensor gradients were bitwise identical between the graph and cached executions, while the gradient after history fan-in into the gated source differed by `1.4901161193847656e-08`. The input gradient of that layer then differed by `2.384185791015625e-07`, and the discrepancy grew upstream. The failed linear weight gradients differed much more than their own fixed-operand FP64 reduction error; changing only weight-gradient reduction was therefore not the indicated fix.

The targeted follow-up widened the shared gated convolution-state edge to FP64 before history gathers and cache fan-out. The finite convolution itself still performed its existing FP32 multiply/reduction. The existing fixed IEEE FFN and fused RMS diagnostic changes were retained; this is a combined diagnostic, not a claim that widening alone fixes the production defaults. All 151 derivative comparisons passed at unchanged tolerances. Every observed forward value and every observed layer/primitive gradient, compared in FP32, was bitwise identical. RNG was preserved. It completed in 86.15 seconds, with 6.4 GiB host peak and 3,855,005,696 bytes CUDA allocation peak under the memory ledger.

Current evidence is bound in `/home/werg/data/dgxops`:

| Artifact | SHA-256 |
| --- | --- |
| `lfm9478-first-adjoint-trace-root-20261018.json` | `313cd5e4a8af81edba7b5d4da1877ad82f23355be3361e8514a36bfcdddf82e0` |
| `lfm9478-wide-conv-adjoint-trace-root-20261018.json` | `5a355f1b648d7bd3c94dcff3e56be695caf6508abe122c2263bcf1aa083df518` |
| `trace-lfm9478-wide-conv-adjoints-root-20261018.py` | `32857fb9889b79d0200bbfb84feaf5deb9dfed67ba182ee658948076ee0f6546` |

An initial 12 GiB invocation failed during CUDA model allocation before any reference execution; its failure is retained separately. The successful expanded traces used a 20 GiB ledger claim. That allocation failure is not a numerical result.

Production integration must preserve the same treatment across graph execution, explicit cached execution, exported/imported differentiable state and checkpoint recomputation. Avoid materializing FP64 history tensors proportional to token count times convolution width when a bounded gather/adjoint implementation can do the work. Bind any changed arithmetic and workspace to the actual parent and capacity identity, restore optimizer/RNG/exposure unchanged, and measure full backward capacity before adopting it. Share the mechanism between models; architectures without finite-convolution state need no artificial convolution path.

The separately attempted cloud FFN chunk transition on checkpoint 12756 failed its own complete 4,099-source-token comparison on 2026-10-18 at 05:05 UTC, before SGD or full policy capacity. The small 9478 result does not supersede that failure, qualify chunk-size changes, or authorize a tolerance waiver. Preserve the captured whole policy graphs as unfinished work while useful established-capacity production training continues.
