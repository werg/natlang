# Native learned-row rates and exact named continuation

The shared native trainer supports a distinct declared rate for existing
Neuralese and reasoning vocabulary rows. Fresh-run defaults remain unchanged.
The supervisor's measured LFM candidate is .003; no Mellum improvement is
inferred from LFM results and no production recipe default is changed here.

SFT and native RL use `--learned-row-learning-rate` (recipe parameter
`learned_row_learning_rate`). The exact selected parameters are
`interface.input_rows`, `interface.output_rows`, `interface.extra_input_rows`
and `interface.extra_output_rows` when present. Tied input/readout rows share
one parameter. The backbone and low-rank embedding adapter retain their
backbone rate. This uses the existing optimizer and
`optim_restore.set_learning_rates`, including its actual child partitions.
Learned vocabulary rows can belong to the ordinary Muon child; they are not
assumed to be AdamW parameters.

On a fresh run, an omitted learned-row rate uses the original constructor rate.
On resume, omission retains each learned row parameter's saved rate; a declared
new rate changes those rates after restoring exact moments. Semantic stages
use the same optional field at the top level, per `models.<participant>`, or in
phase `options` (phase options take precedence). VAE, joint, distillation and
native RL preserve inherited row rates while changing backbone rates.

Raw PortMuonAdamW checkpoints now bind actual serialized child/group slots with
`param_names`. The schema remains a parameter contract inventory; its order
never substitutes for slot identity. Restore preserves exact per-name moments
and group options across group splits/reordering, sparse-row additions and
explicit FP32 promotions. Old unsplit ordinary children and explicitly named
single-parameter latent groups are recoverable from their existing bindings.
An old split group without exact names is rejected rather than guessed or
reset. Component removal keeps the name inventory consistent. Host state
receives owned tensor storage through the existing restoration helper.

Latent in-backward hooks now select their parameter's current group from one
map built per update. Constructor group indices cannot apply another
parameter's settings after a rate split, reorder or named restore. Existing
optimizer/RNG state, row credits and global update counters are retained.
Effective learned-row rates appear in SFT progress and semantic metrics.

Validation through the DGX memory ledger:

- `native-row-rates-final-20261016`: 60 focused tests passed in 109.43 s.
  Includes actual updates on both ports, exact optimizer/RNG continuation,
  actual SFT CLI continuation with row rate .003 and preserved exposure,
  actual VAE/joint/distill shared sharded checkpoint rates, reasoning-row
  additions/FP32 promotions, and existing named optimizer restore checks.
- `native-row-rates-fp8-resume-20261016`: six actual CUDA tests passed in
  NGC Torch 2.14, including exact model/optimizer/backend/RNG next-update
  continuation for both current FP8 and optional BF16-dX modes.
- The reasoning-row test verifier uses actual slot bindings rather than schema
  order, so it checks the named moments after groups change.
- `native-row-rates-quick-20261016`: quick check passed.

The separate root full-check service owns broad integration validation.
These mechanism checks do not claim production task lift on Mellum or that
unconsumed data has received training exposure.
