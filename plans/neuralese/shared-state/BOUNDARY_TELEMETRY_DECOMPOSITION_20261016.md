# Observe learned boundary behavior during existing held evaluation

Historical task traces contain producer spans ending at EOS or their token
ceiling. Sparse parameter updates alone do not prove the closing boundary is
learned. Overall held CE can hide these rare targets.

1. Use the existing protected held graph selection and graph forward. Select
   the model's declared role and reasoning token IDs; do not invent held rows
   or train on evaluation traces.
2. From the existing chunked readout CE, accumulate each selected token's
   supervised count, weight, weighted NLL, correct weight and probability.
   Zero-weight targets remain excluded exactly as in the loss.
3. Aggregate raw sums across complete graphs, then divide by that token's
   weight. Report absent boundaries explicitly with zero support and null
   means. Never infer success from absent targets or use base similarity as
   a gate.
4. Keep the loss and gradients unchanged. There is no extra model forward,
   vocabulary readout, sampling, optimizer step or RNG draw. Training's packed
   loss path is unchanged. These are crisp measurements; the supervisor uses
   them to decide practical recipe changes alongside task and held quality.
5. Share the same observer between both architectures. Verify weighted sums
   against a direct readout oracle and unchanged gradients, plus evaluator
   aggregation and model-mode restoration.

Validation: ledger unit `native-held-boundary-telemetry-20261016` passed all
49 focused tests, including exact loss/gradient equality on both ports,
weighted readout oracle, absent and excluded targets, forward/readout counts
and mode restoration. Quick checks passed. Independent review found no
correctness blocker or additional forward/readout.

`mean_probability` is the weighted gold-token probability only at supervised
held positions whose target is that boundary. It does not measure an
unconditional emission rate, EOS margin or sampled producer-close success.
