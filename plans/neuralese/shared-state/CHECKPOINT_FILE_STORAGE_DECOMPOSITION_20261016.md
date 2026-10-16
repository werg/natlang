# Exact file-backed checkpoint inputs

The supervisor chooses CPU or file-backed checkpoint storage from measured
host/GPU capacity and useful training throughput. Whole long graphs remain
mandatory; storage changes do not justify truncation, changed targets or lost
exposure. File storage is an explicit continuation option, not a default.

1. Save only the tensors the existing nonreentrant layer checkpoint saves.
   Keep its backend precision scopes, recomputation and RNG contract.
2. Copy each logical tensor into one private file mapping, preserving exact
   dtype, shape and values. Flush and release clean page cache instead of
   retaining all saved inputs in anonymous host RAM.
3. Restore each requested tensor to its original device. Multiple backward
   unpacks and higher derivatives may read the same saved file repeatedly.
4. Keep a file alive with its saved-tensor holder; remove it when that holder
   dies. Keep the store alive until every holder dies. Recover abandoned stores
   only after acquiring their exact exclusive lock, never by PID-pattern killing.
5. Inherit the explicit directory through saved SFT and semantic participant
   options; allow explicit switching back to CPU. Bind storage in actual memory
   qualification and record bytes written/read/live/peak in progress.
6. Compare both backbones' outputs, all gradients, next optimizer state and RNG
   with ordinary checkpointing, then qualify actual target whole graphs before
   claiming capacity or enabling the production continuation.

The policy above stays natural language. Crisp code is exact tensor file I/O,
lock/holder lifetimes, CLI and recipe plumbing, immutable capacity identity and
numeric verifiers. Extend the existing checkpoint wrapper; do not fork trainers,
replay graphs, invent a new latent channel or silently change arithmetic.

The shared wrapper, file/lock lifetimes, SFT/RL/preference recipe options,
checkpoint inheritance and CPU switching, semantic participant/phase/resume
settings, capacity identity and numerical-reference plumbing are implemented.
Ledgered final integration passed100 tests (14.61s), including both real native
ports on CPU/CUDA, exact next AdamW moments/weights and RNG, second derivatives,
repeated unpacks, live graph/store lifetime, abandoned-store cleanup, injected
I/O failure, full SFT continuation and VAE/joint/distillation continuation.
Quick main passed. Default storage remains unchanged.

Actual cloud full-model/long-whole-graph capacity and throughput qualification
and independent review remain open. No larger capacity certificate or production
file-storage adoption is claimed. The standing full check covers the published
implementation after the current snapshot completes.

Independent data-agent review found that restoring contiguous values alone
could change stride-sensitive native attention dispatch for external inputs.
The implementation now copies the minimal original storage span and restores
exact shape/strides, including zero-stride expansion and overlapping views.
Offsets are rebased like an ordinary device copy; unused storage prefixes are
not serialized. Counters reflect the copied span, including necessary gaps.
The targeted69-test integration passed13.61s, including10 new CPU/CUDA layout
regressions plus both-native gradient/update/RNG and continuation checks. No
production capacity or throughput claim follows from these local checks.
