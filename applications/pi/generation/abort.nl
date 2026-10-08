---
description: The generation's abort handler (pi-durable generation.ts abort and readCalls, spec §8.5). Cancel a deferred response, answer the calls never started, and end the run aborted.
args:
  facts: PhaseFacts
  checkpoint: GenerationCheckpoint
returns: string
---
This generation was aborted; checkpoint is the phase it was in. In the tools phase this runs only after the round's
tool tasks are terminal.

1. Poll phase: when ai.model(checkpoint.model) is not null, try await ai.cancel(checkpoint.model, checkpoint.handle);
   if it rejects, durable.report(the error text) and go on.
2. Tools phase: the calls never started are checkpoint.pending. message = (await durable.entry(checkpoint.assistant)).model[0];
   for each call ID of checkpoint.pending, in order, take the first toolCall item of message.content with that id
   (skip an ID with none). These are unstarted. Other phases have no unstarted calls.
3. Commit [{ op: "convertPartial" }, then for each unstarted call { op: "appendToolResult", call: { id: call.id, name:
   call.name }, error: { code: "aborted", message: "Tool " + call.name + " was aborted" } }, then
   { op: "endRun", settlement: { status: "unanswered", reason: "aborted" } },
   { op: "next", state: { status: "terminal", outcome: { status: "aborted" } } }].
   A started call whose task faulted keeps its done slot without an entry; the context derivation synthesizes its
   missing result.
Return "aborted" with the number of unstarted calls answered.
