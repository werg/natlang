# Async expert iteration across the G4 and the DGX: continuous collection, back-to-back LoRA rounds

**Status:** v2, the supervising agent's revision of the "Phase 2: G4 + DGX Spark async RL" draft (the draft is commit
dc17fcf68). Every item is decided and required, with no activation triggers or optional stages. It builds on
`plans/neuralese/lfm3opti.md` v3: the LoRA (`lora+rows`) learner core, Colab G4 as a machine, FP8 base GEMMs and the
remote-actor module. Where the two documents disagree on placement, this one wins, and lfm3opti §1/§2/§4/§6 point
here. **Date:** 2026-11-22.

## 0. What the draft got right, and what our conditions change

**Kept from the draft:**
- **Decoupling.** The learner and the collectors run continuously and independently, joined by a durable,
  preemption-proof exchange.
- **Delta shipping.** Only adapter factors cross the WAN, and the merge happens locally on the DGX.
- **Atomic commits.** Data first, a small pointer last.
- **Dedup ledger.** Nothing is trained twice across a resume.
- **The DGX as the durable anchor.** It holds base weights, the pool, the scorer DBs and the archive.

**Changed, because the draft assumed a different method and different economics:**

1. **Our method is expert iteration, not async GRPO.**
   - The learner trains Ambient SFT on a per-task pool of verified trajectories: own best-of-k successes first, then
     teacher trajectories with student-filled reasoning.
   - Nothing in it is an importance-weighted policy gradient. Off-policy staleness and actor quantization therefore
     cannot bias its gradient. They only shift which trajectories get sampled, slightly.
   - So the draft's Pillar 3 (score centering, top-K capture) has nothing to correct here and is **not built** (§6).
     GRPO stays the later polish phase. It is synchronous and on-policy, with its existing TIS.
2. **G4 decode is not idle time.**
   - The G4's advantage over the GB10 is larger for decode (~6x memory bandwidth: 1.6-1.8 TB/s vs 273 GB/s) than for
     training (several times the tensor throughput).
   - A G4 that only trains leaves its strongest capability unused, and the cycle stays bound by GB10 collection.
     With today's numbers, ~6 DGX-hours of collection and ~3-5 DGX-hours of training per cycle, the balance point puts
     roughly two thirds of the G4 on learning and one third on decoding.
   - So the G4 runs the **learner and an actor concurrently** (MPS), and the DGX collects and evaluates full-time.
   - **The A100-80GB session alternates instead** (DECISIONS 2026-11-22 22:00): a round's learner runs alone (77 GiB,
     activation 58) and the actor alone (70 GiB, 128 running) between rounds; `run-sft` stops the session's actor for
     the round and serves it again. Rounds are >= 6 h apart (`--snapshot-min-hours`) and train ~2 h, so the actor
     decodes most of the time either way, and the round, which gates the next policy, runs at the full box.
3. **No cloud bucket.** The DGX is reachable through Tailscale (Pop already uses it; Colab joins it), and the DGX is
   the durable store. A bucket would add credentials, cost and a third copy for no new property.
4. **Continuous rounds replace discrete cycles.** `expert_iteration.py` cycles (6 h collect, then teacher, build, SFT
   and held) become a pipeline: collection never stops, and the G4 trains one SFT round after another on pool
   snapshots.

## 1. Topology

| Machine | Runs continuously | Notes |
|---|---|---|
| **Colab G4** (one or more sessions) | **The learner**: LoRA SFT rounds back to back. **One SGLang actor** under MPS beside it, capped at KV 24 GB and 64 running, with collectors on the DGX. | The learner owns the compute and the actor fills spare bandwidth. Each further G4 session runs an actor at the full running cap. |
| **DGX GB10** | **SGLang actor** at the full cap (128 running), collectors and scorers for **every** actor (DGX and G4), teacher campaign, fill, pool, mix builds, factor merge and publication, **held eval**, Neuralese, operator and diagnostic work | It is the durable anchor. If no G4 is up, the DGX runs the learner itself (`ExecutionProfile` fallback, lfm3opti §1). |
| **Pop** | probes, Pop fill | |

**Held eval runs on the DGX**, in one engine series: SGLang with vendor projection, FP8 KV and BF16 weights, attested.
The DGX is always on, so the held series never depends on a Colab session or on Pop being reachable. Held episodes take
priority over collection on the DGX actor. At 4 samples per task (1024 episodes), an eval uses ~3 h of the DGX actor.
This replaces lfm3opti §4's G4 series.

## 2. The round

A round turns a pool snapshot into a published policy.

1. **Snapshot and build (DGX).** Freeze the pool at the round's cutoff, compute the frontier from the measured pass
   rates, and build the mix (`sft_sources build`, family-scope prefer-unseen, Ambient settings).
   - The build for round n+1 runs while round n trains, so the G4 never waits for data.
2. **Ship the mix (DGX to G4).** Resumable rsync (zstd on the wire: a 1.5 GB chains.pkl crosses in 53 s), ~1 GB of chains. Commit order: blobs, checksums, then
   `round.json`.
3. **Train (G4).**
   - LoRA SFT on the merged start policy of the previous round (merged on the G4 from its base and that round's factors,
     byte-equal to the DGX publication, checked by sha; shipped only when it is not): r=128 on all 9 projection families plus the Neuralese
     rows, FP8 base GEMMs and resident activations.
   - Learner state goes to the DGX every 15 minutes.
   - Expected ~1 h per round.
4. **Ship the factors (G4 to DGX).**
   - The FP32 factors and trainable rows, ~0.5 GB at r=128. For LFM3-1.7B, 24 layers with 11 attention layers give
     ~1.0M factor columns per unit of rank.
   - Plus `policy_manifest.json`, committed last.
5. **Merge and publish (DGX).**
   - Merge in FP32: W = W0 + s·BA. Then apply the trainable rows, round to BF16 once and write the export for every
     engine (lfm3opti §7).
   - **Rebind every actor** (DGX and G4) at call boundaries, with all four routes attested. Each actor's native engine
     reloads too (the lfm3opti §3 fix).
   - The merged policy is the next round's base. Both machines hold it, sha-verified, and the factors restart at B = 0.
6. **Adoption.** The new policy is adopted speculatively, exactly as pipelined cycles are today:
   - the next round trains from it immediately;
   - the actors collect from it;
   - a held eval, on the adaptive interval (a993c0f87, counted in rounds), judges it later. A failure rolls the
     lineage back to the last measured policy and starts the bug hunt (AGENTS.md).

## 3. Collection and the pool

- **Every actor always collects from the newest published policy.** Episodes record the policy sha per call, as
  `filled_reasoning` already does. A rebind lands between calls.
- An episode spanning two adjacent policies is a valid own success for SFT. Its targets are actions the student
  sampled under one of two neighbouring policies, and no ratio is computed.
- **Own-success tier.** The first tier of the mix holds successes drawn by the current or the previous published
  policy. Older successes stay in the pool at the priority the per-task pool logic already gives them.
- **Frontier and selection:**
  - Pass rates update from every scored group, as GRPO's selection does.
  - Collection draws tasks by the frontier: failed first, then unmeasured, then by rate.
  - The teacher campaign targets tasks the newest policy never solves.
  - Fills draw on the tasks the frontier sends to the teacher tier.
- **Dedup ledger.** `round.json` records the episode ids each round's mix holds, and `learner_state` records the
  rounds trained. A resumed learner never retrains a round. A rebuilt mix keeps prefer-unseen over the lineage's
  rounds.
- **Neuralese-declaring task variants run in collection** (`--neuralese-variants`, registry
  natlang-rl-neuralese-variants-20261113: 1,381 variants in 47 families). A selected task that has a variant runs as it
  in half of its groups. Successes enter the pool as own successes, and the held closure is checked per variant.
- **Collection actors use FP8 weights.**
  - SGLang FP8 weights on the DGX and G4 collection actors. The conv states and the LM head stay BF16.
  - LFM3's KV is small: 11 attention layers × 4 KV heads × 128, i.e. 1 KB per token per layer in FP8. At 128 running
    with ~8k contexts, a decode step reads ~6 GB of KV and 3.4 GB of BF16 weights, so FP8 weights cut each step's reads
    by ~18%, more on short contexts.
  - SFT has no importance ratio, so the sampler's precision does not bias training.
  - **The held actor keeps BF16 weights.** The held series measures the model that ships.

## 4. Preemption and failure

- **G4 vanishes mid-round.** Collection on the DGX continues. When a G4 session returns, it resumes the round from the
  last 15-minute state.
- **G4 away more than 30 minutes.** The DGX learner takes over the current round from its last state. The round
  records the profile change.
- **DGX unreachable from the G4.** The learner finishes its round and holds the factors locally until the link
  returns. Nothing is lost, because commits are blob-first.
- **Mid-episode policy changes** happen only at call boundaries, and the per-call sha makes every episode attributable.

## 5. What changes in the code (inventory)

| Component | Change |
|---|---|
| `expert_iteration.py` | Discrete cycles become a round pipeline: snapshot, build, ship, train, ship back, merge, publish and adopt, with held on the adaptive interval counted in rounds. Collection, fill and teacher run as standing services, not stages. Verdicts, reference, rollback and `since_held` carry over. |
| `pop_rl_actor.py` → remote-actor module | Serve, rebind, attest and stop on Pop and Colab, plus MPS start on G4 sessions. |
| `rl/collect` and the shared collector | Multiple actor endpoints, each with its share. Per-call policy sha. Rebind between calls. |
| `train/learner_state.py` | Factor and row export, round ledger, 15-minute state saves to the DGX. |
| Merge and export | `export/native_hf.py` and `tune/export.py` merge in FP32 and write SGLang, native, GGUF (llama-server, wllama) and the `lfm-latest` profiles from one merged policy. |
| `serve/sglang_actor.py` | FP8-weight collection profile (conv states and head in BF16); native-engine reload at rebind; per-route attestation. |
| `rl/held_eval.py` | The series key gets the DGX engine. Held episodes take priority on the DGX actor. |
| Docs | AGENTS.md "Current approach" (rounds, topology) and "Machines" (Colab G4), DECISIONS. |

## 6. Not built, and why

- **Score centering, RSC and top-K logprob capture** (the draft's Pillar 3). Expert iteration's SFT has no
  importance-weighted gradient to correct. For GRPO, §7 removes the mismatch at its source (BF16 actors, a staleness
  bound of 2 with TIS) instead of shipping top-K payloads (§7.3).
- **Staleness windows Δv and their sweep.** No ratio means no bias to bound. The own-success tier (current or previous
  policy) keeps the targets in the student's distribution.
- **Cloud bucket transport.** Tailscale plus the DGX as the store gives the same atomic, preemption-proof properties
  (§0.3).
- **The draft's "G4 colocated fallback" profile.** The DGX is always on; if it were down, nothing would hold the data.
- **SGLang LoRA slots.** A round publishes once an hour. A local merge and in-place reload take seconds.
- **The 200-prompt quantization gate and the FP8-vs-BF16 RL comparison.** FP8 weights on collection actors cannot bias
  SFT. The held series stays BF16, so measurement is untouched.

## 7. GRPO in the same topology (the polish phase)

GRPO stays the method's polish phase (AGENTS.md: "GRPO is the polish phase once SFT gains flatten"). It runs on the
same machines and the same exchange, with these settings decided now.

### 7.1 Placement and flow

- **Learner on the G4.** LoRA r=32 (lfm3opti §5), LR 1e-4 (10x the full-FT 1e-5), `natlang_grpo_v1` (the existing
  objective, unchanged).
- **Actors.** The DGX actor at the full cap, plus the G4 actor under MPS beside the learner and any further G4
  sessions.
- **Collectors and scorers on the DGX**, as in expert iteration.
- **Continuous sampling.** Episodes run continuously across all actors (`--slots` work-stealing as today). A scored
  group ships to the G4 as its token ids, sampled log-probs and decision options (~8 bytes per completion token).
- **Per-step factor shipping.** Each optimizer step ships its factors to the DGX: r=32 is ~130 MB FP32, against
  minute-long steps.
- **Merge and rebind.** The DGX merges in FP32 and rebinds its actor at call boundaries, with all routes and a
  native-engine reload. The G4 actor merges locally from the same factors and attests the same sha.
- **Reference.** The run's start policy is the merged base with B = 0. The reference pass is therefore the bare base
  (adapters off): no reference-weights file and no swap.
- **Recovery.** The learner state (factors, AdamW, RNG, the consumed-group ledger) saves to the DGX every step.
  A resumed learner skips every group already in the ledger.

### 7.2 Off-policy control

- **Staleness.** At most 2 policy versions, the existing `--max-staleness 2`. Older groups are dropped and counted.
- **TIS.** As today: `min(rho, 2)` with the tail floor at `--tail-logp-floor -7`, using the sampled log-probs every
  episode already records.
- **Actor precision.** GRPO actors use **BF16 weights** with the attested FP8 KV, the engine GRPO and the held series
  already use. With staleness ≤2 and matched weights, the ratios stay near 1 and TIS covers the residue, as it does on
  the DGX today.
- **Telemetry.** The ratio statistics and `kl_mean` sentinels already cover this. `rho` per staleness bucket is added
  to `train.jsonl`, so a drift in the stalest groups shows in the step that has it.

### 7.3 Why not score centering with FP8 actors

- **The trade.** Score centering would let GRPO actors use FP8 weights (~18% fewer decode reads, §3) and tolerate
  more staleness. It needs the sampler's top-K distribution for every generated token, shipped to the learner.
- **The cost.** GRPO episodes run ~12k completion tokens, and a step trains 24-48 groups of 8. At K=64 (int32 ids plus
  FP16 log-probs), that is ~1.8 GB per step, ~1 GB compressed.
- **The verdict.** Over a Tailscale path that may be relayed, that costs more wall time per step than FP8 decode
  saves, and the larger staleness buys nothing that more actors don't. Matching weights removes the bias for free.
- **Expert iteration is unaffected.** It keeps FP8 collection actors (§3), because SFT has no ratio.

### 7.4 Switching between phases

- **Same pool.** Expert iteration rounds and GRPO run on one pool and one held series.
- **Entering GRPO.** It starts from the newest held-adopted policy. The driver switches the learner from SFT rounds to
  GRPO when two consecutive measured rounds come out neutral (the "SFT gains flatten" condition).
- **Back to SFT.** It returns to SFT rounds whenever GRPO's own held evals turn neutral twice. GRPO's successes stay
  in the pool as own successes for the next SFT round.

## 8. JOLT+ guided episodes: on, as a pool source in both phases

- **What it is.** JOLT+ guided episodes let the student solve frontier tasks under teacher guidance (`rl/jolt.py`).
  The resulting trajectories are the student's own sampled actions on tasks it cannot solve alone. That is exactly
  the never-solved tier expert iteration wants filled: sft-v8's frontier had 45 never-solved and 124 sometimes-solved
  tasks.
- **Decision.** JOLT+ guided collection runs **on by default** for the frontier's never-solved and lowest-rate tiers,
  on every actor, in both phases:
  - The guidance is removed from the trained context, so the student learns to act without it.
  - The leak filter (`rl/jolt.py guidance_leak`, `grpo.teacher_guidance_leak`; DECISIONS audit of grpo-v4-run1: 6 of 11 guided successes had the
    answer in the guidance) drops leaked episodes before they reach the pool.
  - Guided successes enter the pool tier between own successes and teacher trajectories.
- **The privileged twin's KL term stays off.** The guided successes already deliver the signal as SFT targets and as
  GRPO groups, and adding the twin would count it twice. `grpo.py` already runs without it ("no privileged twin").
- **Code.** `expert_iteration.py` and `grpo.py` collection select JOLT+ guidance by frontier tier. `sft_sources build`
  admits the `jolt-guided` source.
