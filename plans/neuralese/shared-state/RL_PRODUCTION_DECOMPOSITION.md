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
