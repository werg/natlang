# Adversarial self-play and skill improvement

## Goal

Build reusable per-function and per-topic knowledge by playing opponents, reviewing failures, editing skills (including discovery descriptions), and independently measuring whether those edits help. This first implementation improves source-level policies and skill libraries; it does not update model weights online. Admitted trajectories can subsequently train the student through the existing SFT pipeline.

## Implemented surface

- `ts-host/src/self-play/types.ts`: game engine, identified policy snapshot, private decision view, match record.
- `arena.ts`: authoritative transitions, seat-specific observations, finite resource allocation, exact model-free replay.
- `policy.ts`: every player decision runs the same `.nl` program with a frozen skill library in a fresh Node runtime. No cross-seat conversation, network, or code editing. Historical observations must be explicitly provided by the engine.
- `evaluation.ts`: game simulation as a host-owned `SourceCaseExecution`. All players use one recorded executor lane with isolated runtimes. This makes whole-episode offline replay possible using the existing effective-turn recorder.
- `skill-authoring.ts`: custom execution propagates through support search, sealed before/after comparisons, transfer and description/body ablations. Execution identity enters search and confirmation fingerprints.
- Collector/queue accept `--arena-root` for external pinned engine data. Collector retains full matches and per-seat invocation traces in `private-arena-matches/`; these are host-private evidence, not automatic training rows.
- Exporter reconstructs the same frozen arena executor during full episode replay. The existing child rewrite replay, visibility audit and native materializer remain required.
- `scripts/skills/build-adversarial-episodes.mjs`: original scenario packets, one episode per game role, frozen opponents, two distinct support groups and separate query groups. Host profiles and scenario labels stay out of `authorView`.
- `scripts/skills/run-self-play.mjs`: standalone real-model exercise, effective and wire exchanges, private decision traces, exact replay of completed matches, append-only progress. Fresh output directory required.

A case ticket describes a **simulation**, not the literal argument to one player decision. The host invokes `play.nl` repeatedly with the actual private decision JSON. Preserve this distinction in training provenance; never claim a hidden scenario ticket was sufficient input for a player's decision.

## Games and reward provenance

ChessT uses `/home/werg/chesst/chesst-rules.js`, loaded and pinned by SHA-256. It is not copied into Natlang source. Preserve an exact source snapshot in development data when deploying on another machine. The older `/home/werg/CHESST` and replay shard directories are separate sources; they have not been silently mixed into this campaign. Full-state historical teacher records require observation masking before any future conversion.

Original semantic games:

- Clue Intercept: shared private codebook, metaphor clues, receiver and adversarial interceptor, public resolved history.
- Evidence Bluff: choose rhetoric that distracts a reviewer from an explicitly stated claim and dossier. Labels must follow the dossier; the advocate cannot simply redefine truth.
- Meaning Bargain: natural-language private preferences, public offers and messages, exact host-authored utility tables. Utility values represent scenario preferences, not objectively measured psychological judgments. Agreement utility and legality are separate from winning a zero-sum game.

Implemented word games: contextual near-synonyms and taboo clues. Curated clues must map to defensible meanings. An arbitrary cross-product of clue and hidden target is invalid: it rewards unknowable labels instead of semantic interpretation. A challenger can choose a difficult *valid* clue, not an unrelated target.

The initial finite clue/argument menus give reproducible rewards without asking the same model to judge itself. They are a bounded first environment, not evidence that unconstrained natural-language gameplay is solved. Before scaling, add multiple relevant clue choices and longer revealed histories so strategy has a meaningful landscape beyond picking one obvious valid card. Broader free-form games need separately frozen judge specifications, adjudication evidence and uncertainty handling.

## Evaluation and population protocol

1. Freeze engine code, scenarios, opponents, policy files, executor model identity, seeds and resource allocations.
2. Run both baseline and candidate against the same frozen opponent roster and scenarios. Search sees support evidence only; the sealed query never enters the author runtime.
3. Edit only allowed skill files. Description-only changes are valid; target executable remains frozen. Skill-use telemetry diagnoses discovery without rewarding minimum skill count by itself.
4. A legal completed match earns only canonical game scores. Engine/provider failure, cancellation or resource exhaustion has **no invented draw or defeat**. Opponent failure invalidates the comparison. Candidate illegal action is separately recorded as a failed candidate gate; do not award unearned teammate rewards.
5. A candidate must improve support selection and then show positive sealed paired gain with valid gates. A transfer arm, if supplied, must not regress. Raw gains are evidence; a few games do not imply statistical significance.
6. Keep historical opponents and cross-play tests when advancing a population. Snapshot changes occur between episodes, never within the paired evaluation.
7. For repeated population selection, reserve new query source groups or explicitly report adaptive reuse. Do not silently recycle a selection set and call it independent generalization. Retain a protected final evaluation partition.
8. Publication requires exact whole-episode and accepted-child replay, causal visibility, materialization and source review. Completed match replay alone does not admit policy actions to SFT. Never serialize full hidden state into a seat's input. DPO requires identical verified contexts; losses and wins against differing information sets are not automatically pairs.

## Current exercises and outstanding work

2026-10-04: base arena/ChessT/semantic engines passed 26 combined provider-free tests; custom arena execution added three passing tests for private views, request accounting, code/ticket/model identity validation, and opponent infrastructure failure.

The first live exercise (`runs/adversarial-semantic-first-20261004`) exposed missing Node evaluator initialization before any model calls. These four attempts are retained as unscored infrastructure failures. Policy now imports the Node entry point. The second exercise (`runs/adversarial-semantic-second-20261004`) uses the DGX Qwen server through local port 18082, two concurrent matches, semantic games plus a short ChessT king-capture position. Results are still in progress; no improvement or training admission is claimed yet.

Still required:

- Exercise word-game labels and review live failures. Registry, packet builder and runtime pins now include both word games.
- Freeze a new independent runtime containing the arena. Existing v5 generation remains pinned and must not be edited in place.
- Run real adversarial skill revision searches and sealed replay/export; investigate neutral and failed searches before scale.
- Add a richer ChessT scenario catalog: hidden items, forced phases, alternative victory conditions, counterplay and paired colors. The current short tactic tests integration only.
- Expand semantic scenarios and opponent diversity. Implement persisted population generations with revision lineage, exploitability/cross-play reports and reserved source groups before claiming an autonomous league.
- Add dedicated per-seat gameplay SFT admission (exact invocation replay and visibility), separate from already supported authoring trajectory admission.
- Sync source through small Git commits; sync generated artifacts with the existing development-data mirror. Preserve the DGX developer's dirty checkout.

2026-10-04 word-game integration: 36 provider-free tests passed; the packet builder now emits 11 role episodes across five original semantic/word games with 22 distinct scenario groups. The first live Evidence Bluff, Clue Intercept and short ChessT matches completed and replayed exactly. These are integration results, not measured skill gains.

Live follow-up: both word-game matches completed and replayed exactly. Negotiation exposed an observer serialization bug (`message: undefined`); corrected it to omit missing optional values, added persisted replay coverage, and made the arena reject non-JSON views before inference. Exercise runner now saves evidence before replay. Runtime v1 (`runtime-v1-cebe9a9`, seal `9b49d41c732dbaf01ca9a2132068dc54ac36e5bd24b41b406a22108a8432c5fc`) was prepared but must remain unlaunched for collection; use a new corrected v2.
