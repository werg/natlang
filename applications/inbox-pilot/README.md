# Inbox Pilot

Tier 2 Semantic-FP flagship application (`A4`) for triaging incoming email and notifications against standing prose rules.

## Highlights

- **Zero-token crisp fast-path (`pluggable`)**: Routine newsletters and automated receipts settle crisply via transport headers (`List-Unsubscribe`, `List-Id`, `X-Receipt-Id`) without invoking a language model.
- **Prompt-injection boundary (`Untrusted<string>`)**: Message subjects and bodies are fenced with `untrusted(...)` at ingestion and never interpolated into instructions.
- **Standing rules (`withContext` + `compileRule`)**: Prose policies bind onto `decideAction.nl` via `withContext` and compile/verify against labeled examples via `compileRule`.
- **Asymmetric risk gate (`escalate`)**: Escalates to `teacher` whenever local confidence falls below `0.85` or `P(escalate) >= 0.25`; escalated messages are safely held instead of autonomously archived or replied to.
- **Exactly-once effects (`ONCE_EFFECTS`)**: `createOnceMailService` guarantees each `messageId` triggers at most one mutating mail action (`archive`, `saveDraft`, or `hold`).
