---
description: The pi.generation task, one phase at a time. One model request of a run, from preparing the prompt through what the response means, and the tool round it starts (pi-durable generation.ts, spec §8.3).
args:
  facts: PhaseFacts
returns: string
---
When facts.previousAttempt is set, this phase already ran once and failed for that reason: read it first and do not
repeat the mistake.
You carry out one phase of a generation task. checkpoint = facts.task.checkpoint, a GenerationCheckpoint;
facts.task.id is this task, and facts.agent and facts.settings are fixed for this phase. Every branch ends with this task's next state
committed through durable.commit (a list of Op ending with a next operation), by you or by the function you call.
The functions of your folder do their own commits; call them with await and pass what they need.
Return one line saying what was committed. If a commit rejects because this task was aborted or the invocation
ended, stop at once and return "stopped": someone else owns the state now.

When facts.mode is "abort", call abort(facts, facts.task.checkpoint) and return its line. Nothing else applies.

Otherwise first resolve the phase's model:
- prepare: ref = facts.agent.model; request and poll: ref = checkpoint.model; retry and tools need no model.
- info = ai.model(ref) when ref is set. With no ref, the message is "No model is configured"; with a ref that
  ai.model does not know, it is "Model <provider>/<modelId> is not available". Then fail the run and return:
  durable.commit([{ op: "endRun", settlement: { status: "unanswered", reason: "no_model" } },
  { op: "next", state: { status: "terminal", outcome: { status: "failed", error: { message, detail: { reason: "no_model" } } } } }]).

Then by checkpoint.phase:

prepare: return prepare(facts, checkpoint, info).

request: message = request(facts, checkpoint, info). Then decision = classify(facts, checkpoint, message, null). When
decision has committed, classify already committed the task's next state (decision.committed says which): commit
nothing more and return decision.committed. On "answer" call answer(facts, message); on "tools" call
startToolRound(facts, checkpoint, message). Return their line.

retry: durable.sleep(checkpoint.until). Then the next attempt goes back to prepare, so agent and settings changes
made during the wait apply: commit [{ op: "liveGeneration", value: { attempt: checkpoint.attempt + 1 } },
{ op: "next", state: { status: "running", checkpoint: { phase: "prepare", attempt: checkpoint.attempt + 1 } } }],
adding compacted to the new checkpoint when checkpoint has it.

poll: durable.sleep(checkpoint.pollAt). message = ai.poll(checkpoint.model, checkpoint.handle). Then
classify(facts, checkpoint, message, checkpoint.pollAt) and continue exactly as after request. Poll once per phase:
when the message is still deferred, classify commits the next poll; polling again here returns the same message.

tools: this phase runs once every tool task this round waited on is terminal.
- checkpoint.pending is empty: return finishToolRound(facts, checkpoint).
- Otherwise the round is sequential: start its next call. callId = checkpoint.pending[0]. Commit
  [{ op: "createTask", kind: "pi.tool", input: { assistant: checkpoint.assistant, callId }, owner: "self", as: "t" },
  { op: "slot", callId, taskId: "$t" },
  { op: "next", state: { status: "waiting", on: ["$t"], policy: "allSettled", checkpoint: { phase: "tools",
  assistant: checkpoint.assistant, tools: [...checkpoint.tools, "$t"], pending: checkpoint.pending.slice(1) } } }].
