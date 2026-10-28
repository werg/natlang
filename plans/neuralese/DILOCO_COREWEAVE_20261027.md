# DiLoCo on CoreWeave for the native trainers (2026-10-27)

Owner request (2026-10-27): build DiLoCo training in shared abstractions,
following the megatoken implementation, and apply it to train LFM2.5-350M
faster.

## Design

DiLoCo runs N copies of the ordinary native trainer, one per one-GPU sandbox,
each on its own stable shard of source rows. Learners exchange
pseudo-gradients (start weights minus current weights) with a CPU coordinator
over HTTPS every `local_steps` updates; there is no GPU interconnect and no
collective. The coordinator merges a quorum per parameter fragment
(token-weighted, staleness down-weighted) and applies outer Nesterov momentum.
Learners never wait: steps trained while a round is in flight are carried
onto the next global version (overlapped DiLoCo).

Shared pieces, one implementation each:

| Piece | Where | Notes |
| --- | --- | --- |
| Fragment plan, merge, outer Nesterov, safetensors wire | `train/diloco/protocol.py` | Ported from megatoken; durable writes use `common/jsonio.write_bytes_durable` |
| Durable HTTP coordinator | `train/diloco/coordinator.py` | Leases, sequence dedup committed with each fragment, staleness, crash reconciliation |
| Nonblocking learner | `train/diloco/worker.py` | `DilocoLearner.after_step` after each optimizer update; state embeds pending uploads |
| Trainer binding | `train/native_diloco.py` | Identity, `init` from the parent checkpoint, `export` to a continuation checkpoint |
| Trainable set | `native_state_sft.native_trainable_parameters` | The optimizer and DiLoCo use the same list |
| Data shard | `pending_exposure_batches(..., shard=)` | Stable row-id hash (`protocol.data_shard`); live resident appends keep the shard |
| Sandbox service and secrets | `scripts/coreweave/job.py` `service`, `cwctl` share tokens and `push_inputs(secrets=)` | Generic: any job can expose one HTTPS port |
| Stage launcher | `scripts/coreweave/diloco.py` | Composes one coordinator job and N ordinary learner jobs |

Any trainer joins by exposing a `ParameterSurface` (its trainable FP32 tensors)
and calling `after_step`. Mellum uses the same trainer path; its BF16 latents
are their own masters, so installed global values round to BF16 there.

Identity: every learner and the coordinator share the parent checkpoint
SHA-256, the exact trainer argument list (without the `--diloco` block, which
names the coordinator URL), and the DiLoCo settings (run id, learner count,
local steps, wire dtype). Parameter names and shapes must match. A learner
joining at version 0 also requires its loaded parent to equal the
coordinator's seed fragment by fragment.

## Running a stage

1. Recipe: a `native_state_sft` stage may declare `diloco`
   (`{"run_id", "workers", "local_steps", "wire_dtype", ...}`); the launcher
   adds `coordinator_url`. It is an operational parameter.
2. Write a run file (`scripts/coreweave/diloco.py` docstring) naming a
   coordinator job (CPU, holding the parent in its bundle) and the learner job
   (the ordinary trainer continuing from the same parent).
3. `scripts/coreweave/diloco.py launch RUN.json`; `status RUN_ID` reports merges,
   fragment versions and per-learner high-water progress.
4. When learners finish: collect each (`cwctl.py collect`), stop the
   coordinator gracefully (it merges the last rounds), collect its root, then
   `python -m natlang_neuralese.train.native_diloco export --root ROOT --parent P
   --learner-checkpoint W0 ... --out DIR`.

The export is an ordinary full-state rolling checkpoint: global weights,
parent exposure plus every learner's own visits, learner 0's inner optimizer
and RNG (recorded as `options.diloco_export`), and `global_step` advanced by
the learners' total updates. Production training continues from it with
`--continue-from`.

## Lessons from the port (also applied to megatoken)

- `finish` reported a learner whose peers had already finished as
  undelivered. It still waits (bounded) for a pending merge so the steps
  trained after that round are sent; past the deadline a delivered round now
  counts as handed over (the coordinator's close merges it). Handing it over
  immediately dropped those residual steps while peers were alive.
- New global versions were discovered only by the 10 s heartbeat; an upload
  that completes a quorum now fetches the merge immediately.
- Stale-round rejection is a structured 409 detail, not a message match.
- With few rounds, outer momentum 0.9 overshoots and the final single-learner
  flush reapplies accumulated velocity. CPU check: plain averaging reached
  0.087 against 0.069 for one learner taking the same updates (initial 4.2);
  Nesterov 0.7/0.9 over 4 rounds ended at 2.6. Watch the outer settings on
  short stages.

- (2026-10-28) A merge committed by `submit` was reported in heartbeat `versions` only after the next 1 s
  reconcile tick (`published_versions` was refreshed there, not at the merge). The learner whose upload did not
  complete the quorum kept training against its upload point and could not send its next round, so rounds became
  unequal (one learner's 72-step window merged with the other's 8-step one, then the other's 64-step window alone
  at full weight): displacements were summed instead of averaged and the global overshot. The CPU test ended at
  0.23-2.06 (initial 4.20) while both learners were at 0.07; it failed since the port (39e9868a0: 0.376), not a
  later regression. The merge now updates the reported version at once: five runs ended at 0.066-0.075 with
  balanced 8-step rounds (10 merges at 20 ms/step instead of 3-5). Unequal windows can still arise from real
  network delays; per-round window balance is worth watching in the 2-learner CoreWeave smoke.

## Not yet measured

- No CoreWeave run has been made. Learner step time on the assigned GPU, WAN
  transfer per round (LFM: about 1.4 GB FP32 per full sync, 0.7 GB as BF16
  deltas), merge latency and the resulting wall-clock speed-up versus the local
  DGX run (about 49 s/update including eval cadence, an estimate from two log
  timestamps) are open.
- Held loss and task outcome of an export against the same exposure trained
  locally are open; artifacts or merges alone are not capability lift.
- Learner checkpoints live on ephemeral sandbox storage; a replaced learner
  needs the existing private full-state export path before long runs.
