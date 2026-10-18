# Native shared-state RL production decomposition

The policy remains S7 `selectTasks`: typed task/sample choices and the declared
RFT versus GRPO recovery decision. Its natural-language interface stays shared
with the existing controller. Crisp mechanisms enforce provenance, held splits,
atomic state, scheduling and actual optimizer updates.

1. Collect complete natlang trajectories using the native actor and the existing
   collector. Preserve sampled inputs, exact behavior probabilities, native and
   compact-message blocks, producer occurrences and actor checkpoint identity.
2. Admit scored train episodes. Compile their complete invocation graph using
   `native_replay.prepare_objective`; score all sampled tokens in one packed
   graph forward through `native_state_sft.packed_graph_loss`. Infrastructure
   has no reward. Failed flat groups use verified teacher whole-trajectory SFT
   through the existing converted graph recipe; they never invent GRPO signal.
3. Continue the shared native trainer with full backbone/interface optimizer and
   RNG state. A declared objective transition changes the train source and
   objective at a new stage boundary, keeps held separation, and starts local
   exposure from zero. An interrupted stage resumes its exact local cursor.
4. Any weight update invalidates the semantic adapter binding. Refresh the
   existing shared semantic curriculum against updated participants, carrying
   VAE/optimizer state but resetting required phase exposures for changed model
   identities. Perform actual VAE, control/joint and behavioral updates; publish
   only the resulting checkpoint plus atomically bound adapter export.
5. Resume collection only after the updated actor attests the refreshed shared
   participant checkpoint and adapter. Keep sealed packets and per-stage
   exposure for restart; never train twice because publication was interrupted.

No new trainer, state transport or recursive graph replay is introduced. Old
projection/direct stages remain legacy diagnostics. Synchronous sealed rounds
are the implemented baseline; actor/learner overlap is not claimed.

## Existing sandbox bridge

The coordinator sends one exact request artifact to the provider bridge for a
whole sealed local round, or one shared refresh after both learners finish.
The bridge stages verified inputs, starts an owned remote worker, polls it,
and returns the complete SHA-bound result inventory. No remote localhost is
assumed reachable from another machine. Policy/task choice remains S7; these
are outside-world services and atomic-result verifiers.

A stop requests checkpoints: the coordinator writes a request-bound stop file;
the host bridge forwards TERM to its exact remote worker and continues draining
checkpoint/result artifacts. The worker forwards TERM to its exact local
children and waits. Publication waits for all learner and semantic refresh
receipts. A bridge exit code alone proves nothing about completed training.

## Exact actor admission and recoverable controller ownership

S7 continues to decide tasks, training and repairs. Crisp collection admission
checks bind the actual requested participant's four-field native identity and
semantic adapter bytes before and after each sealed collection. A shared
checkpoint file hash alone cannot distinguish its participants. These expected
bindings are part of the immutable collection receipt.

The readiness service retries an exited or unlaunchable child from its preserved
state, with a durable reason and next check no later than 60 seconds; it never
turns failure into successful completion. Completion remains verified by actual
stage/learner receipts, so replaying the controller does not repeat completed
updates. There is no retry-count limit.

Before spawning, the service durably records a unique child ownership token,
explicit argv, working directory and output owner. The token propagates through
its subprocess tree. Restart monitors the exact live process using Linux boot
and process start identity; pidfd signaling prevents recycled-PID mistakes. If
its root disappeared, surviving owned descendants receive checkpoint requests
and are drained before retry. An unobserved exit is resumable failure, never an
inferred success. Historical live PID-only records need explicit draining because
they lack the evidence needed to attach safely.

Deployment binding is crisp: it translates the owner's current two-sandbox
release metadata and admitted curriculum into the existing readiness/bootstrap
interfaces. It makes no task selection or training policy. A terminal trainer
exit is required before transferring optimizer ownership to the shared stage;
unfinished CE is carried through the existing native continuation mechanism.

The first shared curriculum uses the fixed original communication foundation.
Later expanded cohorts belong to the separately declared refresh curriculum;
they never move that first finish line. Deployment preparation accepts an
optional refresh-curriculum input, verifies both source closures and binds the
bootstrap controller to the first curriculum while binding RL refresh to the
second. It adds the actual current base paths and task evaluator to both.
Without that input, the existing one-curriculum behavior is preserved. The
existing raw-row qualification gate still defers untrained expanded pairs to
the last qualified semantic cohort, retaining every pending obligation.


## Complete static execution-task source metadata (2026-10-18)

The task policy stays: teach exact typed returns and composition, then measure
both root modes with the existing actor. Static construction is crisp source
plumbing, not sampled reasoning or a new task selector. Preserve each task's
files, instructions, inputs, expected result and authored root code.

1. Build the ordinary ProgramRecord using the existing definitionProject service.
2. Declare its actual shared Curriculum schema: a single-call, optional-technique
   task with its existing family/shape/split group. All facts are already visible
   in the task, so there are no hidden decisive observations or required edits.
   Arrays are real empty declarations, not an admission fallback for bad records.
3. Declare the three private scaffolds' exact scripted child reference values
   from the independently constructed original constants and child instructions.
   Those source-only oracle witnesses never become model targets or prompts.
   Finish the already computed staged root value through the existing reference
   `reply done` directive; the old one-eval script stopped before result commit.
   Preserve the first eval code byte-for-byte and every expected answer.
4. Call the existing validateCurriculum before emitting each record. Test real
   referenceRow -> unchanged admitRow for every task, including all private
   writer/reader paths; reject changed answers. The old abbreviated descriptors
   threw before admission and are not successful source qualifications.
5. Emit a new immutable source representation/metadata-repair witness against the
   historical repaired12 packet, retaining old failed receipts and pending-live-48
   status. Actual teacher48/root-mode and native student transport remain separate.

No default fields are injected into the admission verifier, no original oracle
is weakened, and no historical task packet or training exposure is rewritten.

Offline reference execution explicitly opts into the existing text-Neuralese
transport for private writer/reader source contracts. `referenceRow` and the
existing authored-world proof share one scripted typed-read adapter. The adapter
returns the exact JSON string in one visible typed opening, preserves escaped
content, and rejects missing or ambiguous openings. It delegates ordinary calls
to the existing source reference driver. The default reference path and executing
model prompts are unchanged. This is text-emulation source qualification with
zero sampled model calls, never evidence of native KV transport qualification.
