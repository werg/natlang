# Mellum task regression: exact-weight MoE prefix control

The cloud warmup is unqualified: the held base reference is 13/16; the
1,200 and 1,600 evaluations both scored 4/16. The 1,600 run had two invalid
actions, five repetition failures and five semantic mismatches, with no
infrastructure or unresolved outcomes. Diagnose this gap before promotion.

The retained local step-292 checkpoint previously failed all three matched
held tasks that its local base control passed. On 2026-10-15, ledger unit
`diag-mellum292-moe-kernels-2416` compared invariant, loop and grouped MoE
kernels on those exact trained weights and three real captured prompt
prefixes. Each forward used fresh caches; norm/RoPE were eager and TF32 was
disabled. The probe ran on local sm121, **not** the cloud sm120 architecture.

| Prompt tokens | Grouped KL relative to invariant | Loop KL | Greedy disagreements, last 16 positions |
|---:|---:|---:|---:|
| 4,946 | 0.000214791 | 0.000159395 | 0 |
| 4,764 | 0.000720753 | 0.000583577 | 0 |
| 4,726 | 0.000427234 | 0.000282258 | 0 |

The unit completed with exit zero. This rules out greedy prefix-token
disagreement at these 48 positions. It does not establish equal sampling,
later feedback/cache behavior, gradients, cloud architecture behavior or
task success. No production kernel changed. Numerical gaps remain measured
facts rather than an explanation dismissed as BF16 noise.

Private exact probe inputs/results are in DGX `dgxops`:
`probe_mellum292_moe_kernels.py`, `run_mellum292_moe_kernels.py`,
`mellum292-moe-kernel-proof.json` and `mellum292-moe-kernels.log`.
The step-292 checkpoint remains assigned to this unresolved regression.
