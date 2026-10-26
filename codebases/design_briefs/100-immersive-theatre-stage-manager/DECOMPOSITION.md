# 100 — Immersive Theatre Stage Manager

## Distinct algorithm: run the show as a cue graph with a separate knowledge ledger

This program is a rehearsal and live-show continuity desk. Its input is a production, the current show packet, source evidence and a request about an upcoming or disrupted moment. It returns a reviewable control proposal. It cannot release a safety hold or represent a cue as executed. Natural-language policy owns the dramatic interpretation, the equivalent nonparticipatory route, the least disruptive safe recovery, and the words in role briefings. Host-controlled acknowledgments, timing measurements, capability checks and SQLite writes stay exact. The production’s independent safety interlock remains outside this proposal interface; no model result or application commit can operate or clear it.

The five production operations do different work over one performance graph; they are not instances of a generic record-cleaning pipeline:

1. `modelDramaturgy` walks scene prerequisites and reveal edges. It maintains separate knowledge sets for each character and audience group, and identifies the dramatic beat that survives a branch. A fact can be known globally but unavailable to a character or audience group. Any path that exposes a secret before its reveal edge is blocked.
2. `modelProduction` joins each proposed scene path against a time-indexed resource ledger: cast, venue zones, props, accessibility provisions, participant boundaries and stage-control roles. It identifies collisions, missing qualification, and which constraint owns the decision. The output distinguishes confirmed, reserved, unavailable and unverified resources.
3. `selectSceneVariants` compares rehearsed alternatives by preserved beat, cast/prop demand, knowledge delta and consent path. It may select only an already approved variant when time is short. A new rewrite is a separate review object with explicit changed dependencies; the model cannot promote it to a rehearsed variant.
4. `coordinatePerformance` topologically orders cues and movement windows from the selected scene route and production ledger. It emits role-specific briefing packets with required acknowledgments. A cue moves through proposed → approved → acknowledged → executed only through host receipts; the language model only prepares the plan and cannot claim execution.
5. `recoverContinuity` replays the actual event sequence from the last accepted show checkpoint, applies the approved variant and logged audience choices, and emits a new continuity checkpoint plus post-show reset obligations. The checkpoint preserves each audience group’s knowledge separately, even when the groups rejoin.

## Interrupts and recovery

A refusal is a final consent state for that moment: choose a materially equivalent quiet route and do not ask again. Cast absence invalidates only variants whose required cast includes that person; a rehearsed, signed-off alternative remains eligible, while a new rewrite requires stage-manager approval. A failed prop is removed from the available-resource ledger; use an approved lighting/sound recovery or stop the affected action. A safety stop enters a latched hold state before any dramatic reasoning: freeze movement and cues under the production protocol; only a named safety-lead all-clear can produce a resumable proposal. No model output can clear the latch.

Source deltas invalidate only scene, variant, cue and knowledge entries that depend on the changed source. A new script or consent boundary cannot silently rewrite a previously executed show record. Every recovery references the prior checkpoint and records the observed path; it is append-only so post-show reset can reconstruct what actually happened.

## Exact boundaries and assurance

Natural language selects meaning and proposes scene routes, recoveries and briefings. Crisp application mechanisms own interval overlap, identity and role checks, versioned persistence and outbox delivery status. External production systems own the physical stop latch and actual cue execution. Natural-language review binds their supplied observations to the current production and signal; deployment must authenticate those sources. The gate for dramaturgy checks reveal prerequisites and audience/character knowledge; production checks current qualification and capacity; variants preserve the beat while staying inside consent and resources; cue plans reject overlapping exclusive resources or unacknowledged critical roles; continuity replays only evidenced events. Missing or contradictory evidence remains a blocker. “Ready for review” never means approved, dispatched or executed.

## Tests retained

The three original finite gate cases keep their pass/fail/needs-review truths and evidence. The five episode fixtures remain the regression set for refusal routing, a signed-off cast substitute, failed-prop recovery, two different audience knowledge states, and a latched safety stop. Representation changes may update paths and types, but not those observations or expected decisions.

## Durable performance checkpoint

The public operation takes immutable `LiveShowSources` separately from `checkpoint: TheatreCheckpoint | null`. Its `resumeState` is the sole carried checkpoint; the manifest binds it for subsequent approved invocations. Source record IDs and candidate/draft status remain source provenance. They never seed an executed-cue receipt. An actual production stage-manager identity is separate from the requested review audience.

## Reviewed execution boundaries

The theatre orchestration is written as natural-language procedures, including route comparison, reveal traversal, per-recipient knowledge, consent, checkpoint authority and recovery. The only new crisp helper performs exact UTC interval comparison. It rejects absent or malformed endpoints instead of equating different window labels with disjoint time. A scene audit consumes a finite verdict and constructs its own typed findings. Planned reveal edges never advance observed knowledge, and route selection compares actual production priorities instead of selecting the first array element. Historical duplicate helper files are excluded from the sole executable source inventory.
