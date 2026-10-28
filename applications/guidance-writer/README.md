# guidance-writer: in-run training guidance

Training users (JOLT+ self-teaching, memetic learning, the learned optimizer) give a run verified facts about its task.
The facts never reach a call as text by default. Instead, inside the same episode, the policy model writes each call's
guidance itself, as Neuralese, and the call imports that state as context after its leading system turn.

```
types.ts                GuidanceFact, GuidanceConsumer
visibility.ts           crisp: which facts a call may see at all (exact rule; rl/privileged_context.applicable verifies it)
crispSelection.ts       crisp side of the selection pluggable: a fixed kind order
selectGuidanceFacts.nl  natural-language side: which visible facts, most useful first
writeGuidance.nl        the guidance: Neuralese<string>, readout: template (one forced producer write)
planTeacherAttempts.nl  how many privileged teacher attempts each scored student group gets (JOLT+ controller)
```

Per consumer call, on its first model turn (ts-host/src/native/training-guidance.ts, run by the native agent):

1. consumer = `{name: call id, role: root | child, text: the call's prompt text}`; keep the visible facts. None: no guidance.
2. Select with `pluggable({crisp, nl}, setting ?? 'nl')`. An NL selection that fails or is not distinct visible ids falls
   back to crisp; the request records the implementation that served.
3. `writeGuidance(consumer, selected)` runs in-process as a child call of the consumer, on the same model and recorder.
   Its result is the written native state. A teacher imports it (`mode: import`); a student only writes it
   (`mode: write-only`, for the OPSD twin). When the writer fails, a teacher gets the selected facts as text
   (`mode: text`, with the reason) and a student gets nothing.
4. Every turn of the consumer carries `x_natlang_privileged_context` (`natlang.guidance-request/1`); every turn of the
   selection and writer calls carries `x_natlang_guidance_policy` `{function, sha256 of the .nl file}` and gets no
   guidance itself. The engine renders the request (rl/privileged_context `guidance_plan`) and the learner re-verifies it.

The run-level control is `x_natlang_guidance` (`natlang.guidance-run/1`, written by rl/jolt.py `guidance_controls`):
`{schema, role: teacher | student, custody, facts, selection?: crisp | nl | shadow, budgetTokens?, writer?: native | off}`; `writer: off` (opt-out, the measurement baseline) gives a teacher the selected facts as text and a student nothing. The collector
hands it to the agent; it is never sent to the model server.

Model-facing text (both `.nl` files) is unmeasured debt in plans/MODEL_FACING_CHANGES.md until sampled live.
