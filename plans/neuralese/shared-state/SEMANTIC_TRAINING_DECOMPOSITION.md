# Executable shared semantic adaptation

The natlang policy is: once any two native models finish their declared
production stages, teach them a shared per-token semantic dialect. First learn
to reconstruct paired native states, then jointly improve producers, write-out
controls and consumers, then distil the improved behavior. Keep going when new
models, admitted pairs or better loss settings arrive. Task results inform the
supervisor; a base-relative distribution change does not forbid improvement.

The recipe declares the phases, complete data exposure, tunable losses and
participant prerequisites. `native_readiness` is the exact verifier and process
plumbing for this policy. It starts with two models and publishes later ready
models without imposing an all-model barrier.

1. Read paired admitted trajectories. Each model has its own native graph,
   authenticated exported payload span, exact decoded payload text/token
   offsets, and explicitly named consuming nodes. Verify train/held separation,
   source groups, tokenizer identity and matching text. Crisp parsing and token
   equality are verifiers, not model policies.
2. Execute each producer graph once. Reuse its result for teacher-forced control
   loss and exported raw native states. Independently prefill the payload text
   to obtain the target-local baseline. Preserve real producer context in the
   reconstruction target; never silently replace it with the baseline.
3. Encode the native states with model-specific adapters and a shared Gaussian
   posterior. Align token spans by exact character overlaps, recording any
   explicitly permitted nearest-token fallback. Decode source codes into target
   state deltas around the target baseline.
4. Replace the declared export imports in each target consumer graph with the
   decoded state. Keep all remaining causal dependencies. Compute consumer
   behavior and producer/control losses using the shared graph objective. In a
   joint phase, gradients reach both producers and consumers through the VAE.
   Defer every optimizer update until the entire joint backward finishes.
5. A distillation phase trains the target model with its own native state to
   match the adapted consumer's detached next-token distributions, together with
   supervised behavior. The VAE is frozen for this phase; no old projection or
   gradient-session replay is involved.
6. Checkpoint all models, the VAE, all optimizers, RNG, exact source identities,
   exposure and phase/data cursors with the existing atomic checkpoint writer.
   A continuation may append admitted data and ready models and change declared
   loss/LR/phase settings. Existing samples cannot change identity under a saved
   cursor. Identity-bound adapter exports distinguish trained checkpoints from
   merely shape-compatible backbones.

The tensor algebra, graph rewriting, immutable identity checks, checkpoint
durability and CLI are crisp mechanisms. Decisions about which new data or
settings improve the system remain supervisor/recipe policy. Completion means
all declared phases covered their full admitted input; a configured short step
limit is an interruption, not a completion receipt. The serving adaptation and
semantic artifact lifecycle must be independently integrated and reviewed
before this is called an end-to-end shared dialect.

Full-model execution proof (2026-10-15): the actual pinned QAD FP32 LFM and
BF16 Mellum completed the joint objective on an authenticated REST27 pair
(6,989 and 7,070 graph tokens). Producer/control and adapted consumer losses,
VAE reconstruction and alignment backpropagated before deferred MuonSR updates
to both backbones and native adapters. Mellum used the production shared
triton-experts backend and activation checkpointing. The ledgered diagnostic
finished in 197.75 seconds, with 77,394,900,992 bytes peak CUDA allocation and
6,288,605,184 bytes peak process RSS. One real zero-width tokenizer offset used
the explicitly reported nearest alignment fallback. The exact source/pair and
gradient/update evidence is in
`/home/werg/data/dgxops/native-real-joint-update-20261015.json`.
This establishes execution and placement for that trajectory, not task lift,
complete recipe exposure, full checkpoint restore or production readiness.
