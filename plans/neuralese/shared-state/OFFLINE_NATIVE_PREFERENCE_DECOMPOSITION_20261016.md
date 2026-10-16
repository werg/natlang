# Offline native preference consumption

## Natural-language training service

`ImproveFromOfflinePreferences` takes admitted, jointly bound source preferences,
a current native participant and its exact optimizer/RNG/exposure state. After
useful shared-state foundation and SFT, it performs adjustable finite preference
continuations, preserves remaining SFT and preference obligations, and returns
the actual updated participant for the existing semantic refresh and RL stages.
Offline source preferences are not samples from the student's policy or a local
correctness oracle. Rejected alternatives never receive ordinary positive-SFT
exposure credit.

1. Authenticate both complete alternatives and the source preference once.
   Apply current source-closed held protection and decision holds to the pair as
   a unit; no held alternative or held shared history enters training.
2. Find the exact common message prefix. Keep it as context, including all
   earlier assistant messages, rather than rewarding common material twice.
3. After divergence, construct each original assistant continuation with its
   exact full native serving history. User/system/tool observations remain
   context. Preserve original thinking at its actual current-turn write-out;
   do not score hidden/dropped historical reasoning as though it were present.
4. Pack both alternatives' continuation invocations into ONE graph forward.
   Sum the teacher-forced log probabilities of actual assistant output tokens
   after divergence and normalize each alternative by its scored token count.
5. Minimize `-logsigmoid(beta * (mean_logp_chosen - mean_logp_rejected) - margin)`.
   Beta is positive and margin is finite, explicitly adjustable. This is the
   declared reference-free offline pairwise objective, not standard reference
   DPO, task reward, base-drift qualification or a second frozen-model barrier.
6. Use the existing shared native trainer, optimizer, atomic checkpoint,
   safe-stop/RNG restoration, row visit/exposure, held/task/runtime evaluation
   and recipe stage mechanisms. Both alternatives constitute one credited row
   only after the real joint update; partial publication retains all deficits.
7. Qualification covers analytical loss/gradients, exact prompt/target masks,
   independent histories in one actual forward, both model ports, full-state
   stop/resume and declared objective admission. Actual combined pair geometry
   needs measured capacity coverage; metadata inheritance alone cannot grant it.

## Crisp mechanism boundaries

- Source/curriculum policy is the contract above and existing shared admission
  policy. No new executing-model prompt or feedback text is needed.
- Pair hashes, common-prefix equality, protected group joins and mask validation
  are exact verifiers of source facts and objective declarations.
- Native rendering, graph packing, forward/backward, Muon/Adam routing,
  checkpoint and continuation are existing shared mechanisms, not another
  training loop or model-specific implementation.
- A preference graph requires its declared objective. Ordinary SFT must fail
  closed rather than turn the rejected alternative into a positive target.
- Stage scheduling is executable through the shared recipe/controller path,
  with explicit remaining obligations carried into later refresh/RL/SFT cycles.


## Implemented qualification and remaining cutover

The first implementation includes the shared trainer objective, explicit recipe
kind, immutable suffix/source bindings, ordinary-CE exclusion, full-state
stop/resume with beta adjustment, carried mandatory pair credits and actual
no-update capacity execution of the combined alternatives. Both tiny native
ports match independent cached loss/all-parameter gradients; actual pinned
Miro trajectories render through both production tokenizers with exact complete
native histories and suffix masks. Capacity proofs bind the declared objective
and its effective beta/margin, not merely a metadata-only size estimate.

Source export/admission, rolling acquired-file snapshots, production curriculum
configuration and actual target-sandbox pair-capacity qualification remain OPEN.
This objective is reference-free offline preference learning, not standard DPO
or an on-policy reward. No production exposure or preference quality lift is
claimed by these mechanism tests.
