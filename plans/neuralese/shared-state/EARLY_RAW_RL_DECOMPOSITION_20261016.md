# Useful raw RL before the shared VAE is ready

The training policy improves each student through comprehensive SFT and useful
full-episode RL as soon as exact trained checkpoints and runtime support exist.
Completing the producer/consumer foundation on both models gates joint VAE work;
it does not gate ordinary or raw-state whole-graph policy updates.

Exact mechanisms to integrate:

1. Read native checkpoint metadata/state through the existing memory-mapped
   loader. Fingerprint its actual embedding path using the same identity function
   as serving. Materialize only the exact tokenizer/role entries, not another
   model. Reject legacy heads, missing optimizer state and mismatched identities.
2. Configure the existing headless native actor and shared native RL/SFT learner
   without a semantic adapter or participant selector for single-model parents.
   Preserve exact actor/checkpoint/archive provenance and entire real graphs.
3. Continue raw updates directly to the same student's current production recipe,
   preserving optimizer/RNG and completed per-row source/foundation exposure.
   A raw update does not require a nonexistent semantic-refresh/export step.
4. Generate executable per-student bootstrap provisioning alongside the existing
   two-model semantic rendezvous. Coordinate safe update-boundary resource
   handoff and resume, without starting forked students or resetting history.
5. Keep adjustable shared semantic policy scheduling explicit in the generated
   curriculum. Semantic episodes remain whole and use the joint participant/VAE
   owner; ordinary/raw episodes are eligible independently of communication.

Policy is this adaptive production instruction and existing task selector.
Identity hashing, tokenizer export, exact checkpoint restore, archive verification,
state publication and provider/resource plumbing are crisp shared mechanisms.
No model-facing prompt, reward or task definition changes are needed.

Implementation frontier (2026-10-16): mechanisms 1–3 are implemented in
`rl/native_raw_policy.py`, the existing production factory/controller and the
shared native trainer. A real tiny-model policy update restores the bare parent
optimizer and RNG; its prior foundation row keeps three of four required visits
in `sft_progress`, and a repeated completed invocation leaves checkpoint bytes
unchanged. Deployment now provisions the adjustable semantic `policy_schedule`
from step 5 without inventing collected sources. Step 4's actual per-student
bootstrap/resource handoff and the complete production rollout/update/SFT cycle
remain outstanding. These changes alone do not mean raw RL is running in cloud.

Successful native batches with no reward spread use positive whole-episode RFT,
including ordinary tasks with no communication. A native all-failure batch uses
the declared verified SFT source. Its controller may publish a finite supervised
checkpoint quantum only when the actual native SFT report records good numerics,
a reached continuation boundary with positive optimizer updates, and the exact
saved checkpoint hash. The report remains `completed:false` with its full recipe
deficits intact. The same predicate prevents replaying an already committed
quantum after a publication failure; it does not accept partial RL/RFT updates.

The existing provider bridge now represents raw rounds explicitly without a
semantic adapter/participant. It returns the actual current, pending supervised,
or stopped partial checkpoint. A published raw update must match the returned
state and carry the exact request parent plus full native training schema,
optimizer and RNG. A stale partial pointer cannot replace an already published
current policy. Semantic refresh requests retain their adapter requirement and
the existing shared participant ownership contract.

Executable raw interleave decomposition (next component)

1. The declared recipe and existing curriculum choose tasks, objectives and safe
   checkpoint quanta. Each student may begin independently after its real original
   optimizer owner drains; shared-dialect readiness is not a raw admission gate.
2. Exact handoff plumbing records the original launch, boot/start/PID and captured
   children, successful terminal receipt, post-drain observations and a separately
   bound full checkpoint. Reject live/recycled/unknown owners, mismatched launch,
   incomplete optimizer/RNG/exposure or changed resident generation.
3. Reuse the existing production factory, whole-episode collector/learner and
   mandatory supervised continuation. One student has one authoritative current
   checkpoint/state. Alternate finite RL and declared supervised quanta across
   every usable recipe stage; retain each stage's deficits and completed credits.
4. Commit each sub-operation before advancing the durable cursor. Restart resumes
   its exact output/parent; it cannot recollect completed policy work or start a
   second optimizer. Actor and provider workers drain before publication.
5. Publish a genuine latest SFT-generated readiness receipt for the actual current
   checkpoint. The shared controller refuses a live raw owner. Once both students
   qualify their entire declared communication cohort, finish the current safe
   operation, release ownership and pass those exact parents to the shared stage.
   All ordinary/pending/preference obligations remain in the inherited checkpoints.
The state machine, immutable binding verifier and process/durability checks are
crisp mechanisms for the already declared adaptive policy, not a second selector
or a fabricated readiness certificate. No new prompts or reward rules are added.

The raw coordinator schedules students independently. A fast student continues
useful recipe/RL cycles while its sibling is still completing its foundation.
After both have actually reported eligibility, it stops resubmitting work,
drains the current operations, and verifies both latest parents before releasing
ownership. Only an actual foundation-qualification operation can set that flag;
completion of an ordinary/preference stage cannot substitute for it.

Resident-at-drain and post-SFT source evidence are immutable snapshots of the
actual worker authority. Later legitimate owner/current-file rewrites leave old
evidence valid. The worker separately checks its live resident owner before
allocation. Exact adopted recipe bytes travel with results; a missing immutable
alias can be restored, while a different existing file is never overwritten.

Focused evidence: real LFM and Mellum four-phase SFT→whole-episode RFT→SFT→RFT
continuations retain global steps 1..4 and communication visits 1,1,2,2. Mellum
runs BF16/bf16x6 with executed current-FP8 projection counters in all four phases
(ledger native-raw-mellum-real-interleave-cuda-20261016: 1 PASS, 8.41 s).
Rewards/actions in these fixtures are diagnostic inputs, not task-lift evidence.
Actual full-size cloud batch geometry/backend qualification and safe operational
handoff remain necessary; no production raw-RL launch is claimed by this commit.
