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

Implementation frontier (2026-10-17): mechanisms 1–3 and the per-role
bootstrap/ownership machinery are implemented. Full-size LFM raw execution has
actually occurred: trained actor step 10407 (`0d781ea562e1d0c664ae11b835296c584c061bc2c84eeb775703722c76158d2e`)
produced three admitted whole graphs and two genuine GRPO updates, saved at
10409 (`6966fee5ab9ab1c9cc0794e2d5506997937f4790a190bc97f3be87d74e4e0d4a`).
The graphs have 44,017 / 9,810 / 43,757 forward tokens and 44,314 / 9,914 /
44,208 compiled KV slots; the executed full-size capacity proof passed with a
14,873,030,144-byte CUDA peak. This is historical execution evidence, not a
claim that a recurring production cycle is currently active.

Exact custody is recorded in
`/home/werg/data/dgxops/lfm10407-to10409-historical-fullsize-raw-execution-audit-20261017.json`
(SHA-256 `11942701145fe492346f906095d0d97fd7cd87a6a4089f8e343d43d8c153ad09`).
It binds the original failed 797 request/result, explicit tool-history repair,
unchanged actor samples/rewards/events, actual graph IDs/bindings, MAX proof,
training report and readiness receipt. The original failed operation remains
failed; the separate owned repair and honest canonical retry account the earned
updates. Later full-size geometries still use the existing actual-source
capacity mechanism when required, rather than inheriting this example blindly.

The remaining recurring-cycle blocker is operational deployment. Read-only live
control snapshots at 22:36 UTC are
`/home/werg/data/dgxops/{lfm,mellum}-current-sourcehost-raw-control-frontier-20261017.json`.
The LF central cursor still names checkpoint 10409, cycle 2/SFT stage 0. The
Mellum cursor retains its already-earned first RL round, while its active plan
still names the older one-stage scaffold recipe. Both actual student lineages
have since earned direct ordinary/preference updates. The active plans do not
yet declare per-source-host ownership/exchange, although staged shared CPU21
code supports it. CPU21 also predates current-source argument normalization.
These are not communication-foundation, VAE-readiness or new-curriculum gates.

Complete the next useful handoff through the existing mechanisms:

1. Prepare source-host controls while the current learner keeps training. Bind
   all source-role arguments from the actual adopted recipe, current qualified
   numerical runtime, whole pending obligations and explicit ordinary 1000 /
   preference 250 update quanta. Retain asynchronous protected task evaluation.
2. At a genuinely useful saved boundary, authenticate the complete owned
   operation chain, including typed failed-monitor saved-update recovery where
   applicable. Decode the final current full parent once on its own source host;
   older closed operations use their exact metadata authority. Preserve every
   optimizer/RNG value and stage/row credit, then CAS the candidate state/cursor.
   Never replay Mellum's earned round or LF's completed repaired collection.
3. Configure the shared per-role coordinator/provider with explicit disjoint
   ownership and `semantic_consumer=mellum`. Keep full parent verification on
   each source host; exchange small authenticated readiness controls. This does
   not assert joint readiness or authorize a second optimizer.
4. Run the existing qualified 140-case TRAIN curriculum through a real current
   actor, full-episode collection/admission and RL/RFT learner, then return to
   every mandatory recipe stage. Twelve newer tasks and completed communication
   foundation are not prerequisites for this early raw cycle. Record actual
   actor/collection/save/restore costs and current checkpoints after execution.

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
Those fixtures alone grant no full-size cloud qualification. The separately bound
historical LF execution above supplies actual full-size evidence for its exact
three graphs; a current recurring-cycle launch and larger geometry remain
separate operational and execution obligations.

Original launch and terminal records are also copied byte-for-byte into immutable
handoff evidence. Their original path/SHA bindings remain explicit for requirement
matching, while later legitimate writes to the operational control slots cannot
invalidate that historical drain proof. A source/snapshot digest mismatch rejects.

Every coordinator failure inside its executor forwards the shared safe stop
before waiting for owned workers to drain, including malformed hot-plan or
storage failures. The original exception is preserved; failure-state writes
cannot replace it with a secondary storage exception.
