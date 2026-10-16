# Native-state SFT decomposition (2026-10-15)

## Policy

The declared `native_state_sft` stage trains the same positional-KV/hybrid-state model used by native inference. Its natural-language policy is: consume every admitted training graph and ordinary-text trajectory under its declared source/split weights; predict each supervised own-token from the actual predecessor query; retain optimizer, RNG, exposure cursor, and exact model state across operational stops and source continuations; evaluate held graphs and real tasks without making base-relative KL a gate. A stage is complete only when the declared full source exposure is complete, not when a short operational update budget ends.

## Typed functions and exact mechanisms

1. `parse_graph` (shared data function) verifies schema, tokenizer identity, split, source groups, imports, and own-token supervision. It is the trust boundary for graph rows.
2. `graph_loss` (shared model function) executes one teacher-forced graph forward and computes predecessor-query token loss. It owns exact graph masks/state placement; the trainer does not replay producer graphs.
3. `native_state_sft` (shared trainer function) owns deterministic source ordering, batching, optimizer updates, finite-gradient checks, exact resume, atomic checkpoint commits, and exposure receipts. These are crisp because optimizer/RNG persistence and token scheduling require exact state.
4. A model-family optimizer factory reuses existing optimizer and Mellum FP8/in-backward policies. The recipe declares settings; it does not branch by family in the objective.
5. Held graph loss and task cadence are evaluation functions over the same restored native checkpoint. They report task quality, held loss, numerics, runtime, and exposure. Similarity/KL from the base is telemetry only.
6. The recipe runner dispatches one typed stage and verifies the report/checkpoint identities. It remains the shared mechanism for both model recipes and continuations.
7. Role-aware conversion/admission is upstream data work: it produces native graph rows with explicit producer/consumer markers, tokenizer provenance, source groups, and protected splits. The trainer must reject ambiguous legacy flat Neuralese text rather than infer roles.

## State and continuation contract

The checkpoint contains the exact `natlang.neuralese-native-checkpoint/1` model state plus optimizer state/schema, Python/NumPy/Torch CPU and CUDA RNG state, deterministic source schedule identity and consumed prefix, update count, and report/held-evaluation state. Adding admitted sources uses an explicit continuation whose prior schedule is an exact prefix; restore optimizer and RNG without resetting progress. Every checkpoint is atomically committed before its report claims the corresponding update. A receipt exposes model/checkpoint identity, source identities and per-source actual supervised-token exposure, held-data binding, optimizer policy, and completed recipe-stage progress for later shared VAE readiness.

Source continuations retain the old row order and semantic hashes as a prefix.
The runner admits an active native stage's new inputs only when
`continued_native_graphs` and, for an expanded held split,
`continued_held_native_graphs` bind the exact predecessor artifacts. The trainer
checks those hashes against its restored checkpoint and checks every prior row.
It schedules each row's remaining required visits from its recorded exposure,
including additions after an earlier epoch completed; completed rows receive
no duplicate visits. Fractional epochs cover the declared source prefix.
Changed packing retains these visit credits and optimizer moments. This exact
accounting is crisp mechanism for the recipe's complete-exposure policy.

## Shared recipe policy

LFM and Mellum inherit one `native_state_sft` stage declaration and objective. Only backbone-inherent optimizer/memory settings may differ. Mellum uses its established MuonSR/in-backward and FP8 memory provisions; an AdamW-only 12B path is invalid. Recipe continuations append newly admitted sources while preserving all already consumed work and optimizer state. Full recipe progression and actual data exposure remain visible to the parent recipe runner.

## Genuine communication before shared-space training

Policy: teach each student its own authenticated producing/consuming graph from
every admitted semantic pair before declaring that student ready for the shared
VAE. Keep all ordinary trajectories and their actual row-visit credits. On an
admitted source append, expose missing communication visits first, then continue
ordinary deficits; never wait for the huge ordinary backlog to finish before
teaching the newly available communication rows.

Exact mechanisms: `NativeGraphSource` accepts an immutable
`natlang.native-graph-sources/1` manifest of native-graph files and semantic-pair
files with selected model names and SHA-256 bindings. It reuses the shared pair
verifier, checks actual model role IDs and consumer supervision, and indexes the
selected whole graph without retaining token arrays. Prior ordinary row hashes
and source order remain an unchanged prefix. Communication priority changes
execution order only; requested indices must yield exactly that order even for
compressed inputs. The durable authority is actual per-row optimizer-update
credits, preserving RNG and moments across source append, partial stops and
repacking. Report and readiness contain matching `communication_exposure`
{graphs, required_graph_visits, graph_visits, complete}; ordinary-only training
cannot declare readiness for the VAE. These verifiers, streaming mechanics and
exact accounting are crisp; the exposure-first policy is stated above.

Cadence held-loss evaluation may rotate through whole held graphs under declared
`held_eval_tokens` / `held_eval_graphs`; oversized graphs are evaluated intact.
Coverage cursor, actual graph/token visits, elapsed time and scope are saved in
checkpoint metrics. This is an operational/resumable cadence setting. Final
completion always evaluates the full held source. Training data and graph
lengths remain uncapped. CUDA progress records allocated/reserved bytes and
lifetime peak allocation without synchronization or resetting the peak.

Native source manifests recursively bind every referenced graph/pair file in
recipe input identities, decoder checks and adaptive handoff filtering. Relative
references stay portable; the tiny manifest is read in place so moving it alone
to an HDD cache cannot break its relative paths. Declared predecessor hashes
are remembered after adoption, allowing subsequent restarts with the same
continuation declaration. An exact known previous ordinary-index policy may be
reused only with unchanged parser/compiler hashes and verified metadata; every
streamed graph is rechecked against its immutable row binding before training.

Verified integration: 86 focused source/trainer/recipe checks pass, including
actual hybrid LFM and Mellum one-flat-forward consumer-only gradients into the
producer adapter, independently cached capture/splice parity, exact requested
stream ordering, stop/append/priority/repack/resume and unchanged completed-run
RNG state. The first admitted corrected REST shard passes strict extraction for
both actual model IDs: 37 positive communication graphs/model; this validation
records zero training visits and does not claim full REST coverage. Empty
source-authenticated payloads remain trainable structural examples and receive
no genuine communication credit. Cadence checks preserve complete long held
graphs and always require a full held-source pass before completion.

Explicit source `training_priority` is a nonnegative integer scheduling policy
input in `natlang.native-graph-sources/1` (default zero; omitted enclosing values
retain a nested source declaration). The manifest SHA binds policy changes.
Communication deficits execute first, then higher source priority, then stable
physical row order within each visit. This changes only the pending optimizer
schedule: row bindings, quotas, token weights, epochs and completed credits stay
unchanged. The existing exact-index streamer supplies the requested order.
# Active communication-foundation handoff

The declared recipe selects a finite admitted communication cohort while
ordinary CE remains mandatory. The crisp handoff verifier accepts its initial
bound input and boolean boundary setting without resetting row credits; later
appends retain that exact cohort. At first activation the existing indexed
communication IDs and row bindings must equal all communication rows currently
declared to the trainer. A pair-only cohort may omit ordinary rows, but may not
silently weaken the declared communication coverage. This is an exact source
and continuation verifier, not a second training scheduler.


Explicit actual-backend qualification is exact verification, not a new training
policy. The recipe may request `qualify_training_backend`; after restoring the
same model/optimizer/RNG, `eval.native_backend` calls the existing cached-versus-
graph forward/consumer-gradient/incremental verifier inside the actual training
backend scope, requires nonzero executed projection counters and unchanged
numerical tolerances, then restores RNG/backend state/counters. A passed check
records no optimizer update and grants no task/foundation qualification. Useful
subsequent updates provide separate finite-loss and execution telemetry. Disabled
backend/missing-tolerance requests fail before expensive source/model loading.

## Preserve completed updates when held evaluation fails (2026-10-16)

The supervisor decides whether a failed evaluation needs repair or a restart.
The trainer supplies exact durability: after a successful optimizer update and
row-exposure accounting, a held-cadence exception saves that completed state
through the existing atomic checkpoint writer, restores pre-evaluation RNG,
and re-raises the original failure. Failed evaluation does not qualify the
weights or overwrite prior held/task evidence as current evidence. Errors in
the optimizer update never enter this recovery path because in-backward updates
may have partially changed parameters. Checkpoint-write failure preserves the
previous durable slot and is attached to the original evaluation exception.
This is crisp exception/durability plumbing, not another recovery policy.
