---
description: The generation's request phase (pi-durable generation.ts request and streamResponse, spec §7.2, §8.3). Committed context through the cutoff in, the provider's terminal message out.
args:
  facts: PhaseFacts
  checkpoint: GenerationCheckpoint
  info: ModelInfo
uses: [harness/context]
returns: AssistantMessage
---
Send this generation's request. checkpoint is { phase: "request", attempt, compacted?, model, thinkingLevel,
streamOptions, cutoff }, pinned by prepare; info describes checkpoint.model. Recovery reruns this phase with the same
committed messages and options, so do exactly these steps, all in one eval, ending with `return message;` and finish
true.

1. Start the attempt: durable.commit([{ op: "convertPartial" }, { op: "liveGeneration", value: { attempt:
   checkpoint.attempt } }]). This turns a partial answer left by an interrupted attempt into an aborted assistant entry
   and shows the new attempt. The list has no next: the state stays as it is. (A rejected commit means the task was
   aborted; its error ends this call, which is right.)
2. messages = (await context(checkpoint.cutoff)).messages: the committed context through the cutoff.
3. The beforeRequest chain. names = await durable.hooks("beforeRequest"). For each index i of names, in order:
   r = await durable.hook("beforeRequest", i, [{ messages }]); when r.value has a messages list, it replaces messages
   for the next handler and for this request only. An error result changes nothing.
4. message = await ai.turn(checkpoint.model, messages, { ...checkpoint.streamOptions, thinkingLevel:
   checkpoint.thinkingLevel, sessionId: facts.sessionId }, { attempt: checkpoint.attempt }). The host streams the
   partial answer into pi.live while it runs.
5. Return message unchanged: store nothing yourself; classify decides what it means.
