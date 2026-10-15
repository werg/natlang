# Actual full-size joint native CUDA fit

On 2026-10-15 the existing local pinned QAD LFM350M (FP32/IEEE) and
Mellum12B (BF16) completed the production semantic and policy objectives together
on the DGX GB10. This is architecture/kernel/memory/gradient evidence, not a
production foundation certificate, current trained-weight qualification or
model-quality claim. Both cloud trainers and the active historical LF2750 task
evaluation were preserved.

The diagnostic used complete admitted REST27 train row
`e2e-edge:600f51fc88e47b8197383f1e53fa93df894466913cc4c1ef9434fb098851f9d9`:
6,958 LFM tokens and 7,041 Mellum tokens, with no graph truncation. Source SHA is
`6ec7db27f849afdaf43f45fed23871eac8be87b9873c76e42a640b711d0f20e4`;
row SHA is `d4270cb15390a7e8f18a9cb2b01fa766f95fbc6d0eb38a27d7dff91e9901b112`.
The runner verifies the source hash and exact original row, model inventories
and tokenizer contracts. Its in-memory tensor-content identities are explicitly
not checkpoint-file hashes.

Shared production code performed the work: the native factory and FP32 sparse
reasoning/role rows; Mellum triton-experts; deferred shared MuonSR optimizers;
SemanticStateVAE(128,512); `joint_loss`; actual `native_step`/`splice_block`;
existing masked sampler/replay; `NativePolicyObjective` and `packed_graph_loss`.
No alternative trainer, projection channel, synthetic readiness receipt or
50GB checkpoint export was used.

| Actual update | Loss | Finite gradient norm | Model role rows changed |
|---|---:|---:|---|
| VAE | 2.12946 | 1.38563 | Neither, as required |
| Joint producer controls and adapted consumer | 11.31315 | 312.12607 | Both |
| Behavioral distillation plus controls | 3.82761 | 126.68994 | Both |

The subsequent native policy check on each resulting full-size model used a
forced writer with zero sampled actions and an actual sampled reader. It
retained raw positional KV/hybrid state and exact normalized sampling masks.
Two explicitly controlled opposite outcomes exercised GRPO; these are numerical
fixture outcomes, not environment task rewards. Both existing optimizers
continued from the preceding semantic updates. Both native interfaces actually
changed after backward/step. LFM adapter gradient norm was 0.78091 and Mellum's
0.92732. Captured/replay maximum log-probability differences were respectively
1.91e-6 and zero; initial ratio clipping fraction was zero in both. Mellum's
zero scalar GRPO loss did not mean zero gradient: opposite signed on-policy
terms cancel their value while distinct sampled actions retain a nonzero
parameter derivative.

Total runtime was 408.3 seconds, including approximately 187 seconds to load/hash
both real models. Peak CUDA allocation was 77,749,777,920 bytes (72.41 GiB),
reserved 79,989,571,584 bytes (74.50 GiB), under an 82 GiB ledger claim with 2 GiB
reserve. The active historical task evaluator remained running. This establishes
one genuine 7k+7k graph's fit, not an upper bound for all production graph shapes.
One cross-tokenizer alignment direction used the recipe's declared nearest
fallback for one target token; cross-tokenizer state equivalence is not assumed.

Durable evidence is under
`/home/werg/data/dgxops/native-joint-cuda-fit-20261015/run/`: `proof.json`,
`progress.json` and `source-bindings.json`. The latter binds exact script bytes
and the clean tracked shared-code commit. The runner is
`scripts/neuralese_data/qualify_native_joint_cuda.py`; use its explicit input
paths and `--help`, under the memory ledger, when another real architecture
concern warrants repeating it. The 2 tiny canonical-port tests qualify only the
runner's trace plumbing; they are not the full-size CUDA evidence.

This did not test current cloud-trained parents, full protected task quality,
production foundation exposure, participant-sharded shared save/reload or the
complete distributed provider cycle on cloud GPUs. Those retain their existing
separate readiness/transport and exact checkpoint qualification requirements.
