# Native generation progress observation

The supervising natlang process observes actual work before deciding whether to
investigate, continue or interrupt it. Progress counters are evidence, not task
success, graph qualification or a new termination policy.

1. The existing generation scheduler increments one integer after each native
   generated-token forward completes, including private produce-span tokens.
   The counter never inspects, renders or synchronizes tensor values.
2. The existing HTTP service exposes that counter and approximate scheduler
   inventory as Prometheus text. The read does not acquire the actor execution
   lock, so a busy model remains observable. No text or hidden state is exposed.
3. The queue observer polls the supplied loopback server's origin, retaining
   existing llama.cpp compatibility. An unreachable or unsupported endpoint
   means unknown progress; it does not prove a stall. External provider servers
   do not gain unauthenticated metrics requests.
4. Durable reply accounting includes current partial or finished artifacts once
   per exact task identity, but excludes authored reference actions explicitly
   marked unsampled. Those are saved actions, not actual model replies.

All additions are crisp exact plumbing/measurement. Existing natural-language
supervision policy and structural execution semantics remain unchanged. This
adds no model-facing prompt, token, task, reward or error wording. Shared native
serving covers both student architectures; no new model-specific runner is used.
