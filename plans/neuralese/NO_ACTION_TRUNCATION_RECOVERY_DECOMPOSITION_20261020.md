# Completed replies that never execute an action

`continueOrRepair(actualInvocation, completedReplies, executedProgress, declaredLimits)`
is the existing execution/collection policy: allow useful long calls and actual
recovery, but a completed truncated reply with no executable action is a failed
continuation. Reuse the explicitly configured repair/identical-turn limits; add
no default budget or new instruction. This repairs observed repeated 8192-token
no-action replies, not a transport stall or a claim that long reasoning is bad.

1. The native invocation owns repair decisions. Count completed truncated replies
   in its existing failure-repair budget. Preserve the existing cut-off recovery
   sentence. A real successful execution/result may reset failure history under
   the existing law; malformed/unexecuted proposals do not. A pending decode has
   not completed a reply and cannot increment this count.
2. The existing collector observes durable actual model replies. Authenticate the
   partial's program/digest and scope the streak to its latest actual invocation.
   The configured identical-turn bound also bounds repeated completed no-action
   truncations, including differing text, without requiring a tool observation.
   Preserve the original exact call/observation repeat verifier independently.
3. Crisp progress custody reuses the runtime's live `traceFor` action events:
   optional content-free successful-action count/sequence goes into the partial
   record, never the model prompt or wire request. Only increased real execution
   resets the new collector count. Older partials support the strictly provable
   contiguous truncated/no-call suffix; do not invent missing progress receipts.
4. Ended no-action continuation is explicit `model_incomplete`, distinct from
   infrastructure failures, pending work and successful accepted returns. Retain
   original sampled replies, invocation IDs, request hashes and terminal evidence.
5. Check both native execution and the actual queue child lifecycle with synthetic
   replies: recovery, genuine progress, failed proposals, nested invocations,
   pending decode and immutable historical results. Existing 19006 stays frozen;
   no protected task/answer is admitted into training or any fixture.

Policy is the existing invocation/collection recovery contract. Crisp code only
counts exact completed receipts, verifies scope, records durable metadata and
terminates the explicitly owned collector child. No alternate agent/trainer,
new sandbox, inferred semantic success or mandatory Neuralese edge is introduced.
