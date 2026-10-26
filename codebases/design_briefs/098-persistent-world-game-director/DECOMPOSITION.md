# 098 — Persistent World Game Director

## Source-derived procedure

Five domain stages: interpretIntent, modelCharacterViews, chooseWorldActions, advanceConsequences, maintainContinuity. Each stage has a bounded evidence preparation step, field-level decision, typed draft construction, and independent audit. Ten cross-cutting procedures frame scope, select authority, plan dependencies and evidence, reconcile collections, resolve questions, propose bounded repairs, prepare review, compose a recipient-specific view, and assess the impact of source deltas.

## Domain algorithms and gates

- **interpretIntent** — Interpret player language in the visible world context and map it to permitted action types. Distinguish speech, intention, and executed action, ask only consequential clarifications, and avoid granting capabilities because a player narrates success confidently. Gate: Intent resolves to a legal action proposal or remains unexecuted.
- **modelCharacterViews** — Build each character’s view from witnessed events, received messages, memory, and approved world facts. Separate true facts from beliefs and secrets, preserve obligations and motivations, and prevent the narrator’s global knowledge from leaking into character decisions. Gate: Character-visible context contains only authorized knowledge.
- **chooseWorldActions** — Choose among exact legal actions in a way consistent with the character’s knowledge, goals, and relationships. Generate dialogue and creative alternatives within world and content boundaries, and avoid overriding resource or location constraints for narrative convenience. Gate: Action is legal and dialogue does not reveal unknown or restricted facts.
- **advanceConsequences** — Apply actions through the exact world simulator, then interpret their social and narrative consequences without rewriting physical outcomes. Preserve causal history, update beliefs only through information channels, and record unresolved story obligations for later sessions. Gate: Resource conservation, chronology, and knowledge propagation remain valid.
- **maintainContinuity** — Review long-term consistency and introduce events that develop existing tensions or obligations. Check that new events are possible from current state, respect participant boundaries, and preserve stable checkpoints for replay, moderation, and recovery. Gate: No retroactive state invention or unapproved content-boundary change.

## Action boundary

The natural-language program produces a proposal for accountable review. Authorization, persistence, and external effects are exact host responsibilities. Evidence gaps and conflicts stay visible; no action is inferred from fluent completion.
