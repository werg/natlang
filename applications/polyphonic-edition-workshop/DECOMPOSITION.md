# Polyphonic Edition Workshop

## Task and closed source semantics

Reconstruct an ensemble performance from a directory of conflicting partbooks, copying histories,
scoped notation conventions, editorial slips and rehearsal observations. Produce all materially
supported editions, each with a source-grounded event timeline, and explain both rejected readings
and surviving ambiguity. A later copy is evidence of what its exemplar contained; it is not an
independent vote. An editor's authority to correct a location does not establish which ambiguous
location they meant.

This is a new application family. The first world uses explicitly fictional notation. Every duration,
pitch operation, accidental scope, transposition, repeat and correction rule is supplied in the
fixture. Familiar musical terminology supplies no hidden convention. Original evidence comes before
interpretation; the application must not start from an expected score or a preferred edition.

The source tree is planned as approximately 27 meaningful NL callables. Numbered NL owns source role,
copy dependence, correction authority and target scope, notation meaning, competing readings,
performance occurrence expansion and uncertainty. Exact code owns file I/O, original quote binding,
integer tick arithmetic, immutable frontier bookkeeping and external certificate verification.
There is no TypeScript decision implementation.

## First connected world: Glass Harbor

Two voices, Bell and Reed, perform Dock, Channel, then Dock again. Each section occupies four beats.
Source pitch ranks are integers in a fictional scale. Reed sounds two ranks above its written rank;
Bell is untransposed. Three written duration signs mean one, two and four beats. A local raise sign
persists for the same written rank within a voice and section; a fresh section occurrence resets it.
The reprise reuses the same written events but has new occurrence identities and fresh notation state.

An autograph and two dependent copies preserve different readings of one Dock duration. The second
copy repeats the first copy's long reading, so the two copies form one evidence lineage. A signed
editorial slip explicitly changes that duration, and independent rehearsal synchronization supports
the correction. This join requires authority, target identity and cross-voice timing; counting copies
would yield the wrong edition.

A separate authenticated Channel correction says to lower the final entry in one of two adjacent
parts, but its attachment cannot be resolved. Both targets fit the surviving source description and
the rehearsal sonority. A later copy silently chooses one target; its expansion is explicitly
unconfirmed. The application must preserve two editions and the same ambiguity on their repeated
source structures without multiplying independent performance choices.

Negative evidence matters: an unsigned player pencil note is a proposed fingering aid, a copied
rehearsal summary is not another observation, the signed duration repair does not authorize a pitch
change, and a rehearsal statement about simultaneous sounds does not identify their voice owners.
A control packet asks for the different source readings, dependency families, timing constraints,
notation-state transitions, all supported editions and a compact ambiguity certificate.

## Typed state and ownership

| Typed object | Meaning and owner |
|---|---|
| `SourceClause` | Original source ID/path/text and exact quote; reader owns identity binding. |
| `WitnessRole` | Original/autograph/copy/editorial/rehearsal/convention role, scope and certainty; NL source-role child. |
| `CopyLink` | Directed exemplar-to-copy relation with scope and witnesses; dependency reader. |
| `EvidenceFamily` | Original witnesses connected by actual copying, with independent observations retained separately; family reducer. |
| `NotationProtocol` | Supplied duration lexicon, pitch operations, per-voice transposition, reset and repeat policies; convention interpreter. |
| `WrittenEvent` | Source location/voice/section/order plus literal notation tokens and witnesses; partbook unit reader. |
| `EditorialClaim` | Actual requested operation, authority, original target candidates and source attachment; editorial interpreter. |
| `TargetDecision` | Exact target relation and applicability true/false/unknown with original evidence; scoped NL target child. |
| `ReadingBundle` | Alternative source readings and justified corrections for one location; arbitration owns admissibility. |
| `EditionHypothesis` | One consistent assignment of ambiguous editorial targets and source readings, not one assignment per performance occurrence. |
| `NotationState` | Current section occurrence, voice-local accidental map, previous sounded event and elapsed ticks. |
| `EventMeaning` | One actual notation child result: duration, pitch effect, sounding/continuation meaning and state delta. |
| `PerformedEvent` | Original written identity plus fresh occurrence ID, onset/duration/sounding rank and consumed state/evidence. |
| `TimingConstraint` | Source-interpreted relation across events/voices, with exact tick relation and scope. |
| `ConstraintFinding` | Satisfaction/conflict/unknown after joining actual performed events, not a raw-source answer reconstruction. |
| `EditionCertificate` | Actual surviving timelines, rejected hypotheses with reasons, dependency families, coverage and irreducible ambiguity. |

Generated hypothesis/occurrence IDs may be consistently alpha-renamed. Original source, written-event,
voice and section IDs are exact. Timelines have chronological order; evidence/family collections are
sets. When the notation source leaves a reading unknown, retain typed uncertainty rather than a
numeric sentinel that could accidentally satisfy a timing constraint.

## Callable folder graph

The graph has distinct reader branches, a dependency reducer, a hypothesis reducer and a stateful
performance interpreter. It is not one uniform chain. Each listed node creates an owned contribution.

| Callable path (relative to root) | New NL contribution and consumer |
|---|---|
| `reconstructEdition` | Compose actual source archive, arbitrated readings, performed hypotheses and certificate. |
| `reconstructEdition/readArchive` | Directory snapshot, original clause coverage and actual reader branch results. |
| `readArchive/classifyWitnesses` | Source role, authorial/rehearsal/copy distinction and explicit scope. |
| `readArchive/readConventions` | Interpret the supplied notation and edition-admission policies into typed protocol. |
| `readArchive/readPartbooks` | Finite source-location inventory and typed retained per-location readings. |
| `readPartbooks/advanceLocation` | Consume one original reading unit, append actual event or unresolved issue. |
| `advanceLocation/interpretNotation` | Scoped NL literal reads one diplomatic notation statement without performing it. |
| `readArchive/readEditorialClaims` | Separate authorized changes, proposed aids, unresolved attachments and documentary claims. |
| `readEditorialClaims/judgeAuthority` | Scoped literal returns authority/scope from original issuer and source policy. |
| `readArchive/readRehearsal` | Interpret observation scope, event identity and independent timing/sonority constraints. |
| `reconstructEdition/buildFamilies` | Reduce actual copy links into scoped evidence families, preserving independent observations. |
| `buildFamilies/advanceFamily` | Add one source dependency; preserve scope restrictions and expose cycles/conflicts. |
| `reconstructEdition/arbitrateReadings` | Own finite original written-location frontier and source reading alternatives. |
| `arbitrateReadings/advanceReading` | Join one location's source readings, authority and evidence families. |
| `advanceReading/resolveTargets` | Actual scoped NL target decisions over one claim and original location candidates. |
| `advanceReading/compareSupport` | Distinguish independent support, derivative repetition and authoritative correction. |
| `reconstructEdition/assembleEditions` | Enumerate finite compatible reading/target assignments with global claim consistency. |
| `assembleEditions/advanceHypothesis` | Extend one partial hypothesis from the actual next reading bundle. |
| `advanceHypothesis/checkClaimConsistency` | Preserve one claim's target choice across all related locations and future occurrences. |
| `reconstructEdition/performEditions` | Drive each actual hypothesis through its source-derived performance occurrence order. |
| `performEditions/advanceOccurrence` | Initialize reset semantics, gather actual voice events and retain completed occurrence state. |
| `advanceOccurrence/advanceVoiceEvent` | Consume one written event, current state and interpreted protocol. |
| `advanceVoiceEvent/judgeEventEffect` | Scoped NL literal determines actual duration/pitch/state effect; parent binds exact arithmetic. |
| `reconstructEdition/checkEnsemble` | Join actual timelines to independent rehearsal and section-duration constraints. |
| `checkEnsemble/judgeConstraint` | Scoped literal interprets evidence relevance and relation ownership before exact numeric check. |
| `reconstructEdition/explainAlternatives` | Compare actual surviving hypotheses, identify unresolved editorial distinctions and supporting evidence. |
| `reconstructEdition/auditEdition` | Check actual source coverage, decision provenance, state continuity and returned certificate completeness. |

Paths in the implementation will preserve immediate callable visibility. Sibling orchestration stays
in the common parent; a nested child receives its full typed semantic input instead of reaching for
an unrelated sibling or reconstructing it from raw documents.

## Changing-state algorithms

### Reading-unit reduction

1. Derive original reading units from the archive; units contain source IDs and locations, not answers.
2. Keep `{pendingUnits, completedReadings, unresolved}`. A named step consumes its first actual unit.
3. Await the unit's scoped interpretation, bind exact original quote identity, retain the actual result.
4. A failed or unresolved unit keeps its ID and reason. Previously completed readings remain intact.
5. Terminate by strictly decreasing `pendingUnits.length`; compare completed original IDs to inventory.

### Consistent edition frontier

1. Start with one empty hypothesis and the finite ordered original reading bundles.
2. A named step extends each live partial hypothesis only with alternatives in the next actual bundle.
3. Consume NL claim-consistency judgments before accepting an extension. An editorial slip choosing
   one target cannot also choose its mutually exclusive target at another location.
4. Rejections retain their actual contradiction and witnesses. Missing evidence stays an unresolved
   branch, not an arbitrary choice. Consume one original bundle per step.
5. The measure is remaining original bundles, not the number of hypotheses, which can increase.

### Performance-state reduction

1. Expand the source-defined occurrence order once from the typed protocol and selected hypothesis.
2. Each occurrence receives fresh voice-local notation state as the source reset policy requires.
3. Visit each voice's finite ordered written events; one scoped NL effect judgment consumes that
   event, the current notation state and actual protocol. It returns a typed duration/pitch/state delta.
4. Exact arithmetic computes onset and end from that actual judgment. Apply only its returned state
   update and append the actual performed event, preserving source and occurrence identities.
5. Process the next original event/occurrence by explicit cursor. Static iterateOn arguments remain
   context; the changing state chooses the next item. Coverage counts source-event/occurrence pairs.

The per-event frontier decreases by remaining events; the outer occurrence frontier decreases by
remaining occurrences. Source repetition is expanded from a finite written performance instruction.
No heuristic budgets or wall-clock limits define musical completion.

## Exact seams and independent evidence

- Folder reads and manifest/path containment use existing runtime services.
- Exact quote substring and original ID checks bind provenance after the NL judgment.
- Integer duration sums, transposition addition and onset arithmetic operate on supplied semantic
  values. They do not decide which glyph means a duration or which correction applies.
- External verifier checks source identity domains, source-event coverage per occurrence, exact tick
  continuity, voice/section totals, parent-to-child state continuity and allowed report alternatives.
- Independent reference is outside model-visible roots. It records both surviving editions, rejected
  source alternatives and the reason a copied majority is not independent evidence.
- Positive controls include row/evidence permutations and consistent generated-ID renaming. Negative
  controls include treating a copy as independent, applying the wrong correction scope, carrying an
  accidental across a reset, shifting the transposing voice twice, omitting one supported edition,
  choosing both targets of one editorial slip and replacing uncertainty with a convenient pitch.

The exact checker validates declared finite invariants and replay, not arbitrary prose entailment.
Peer review reconstructs the source world independently. Native loading and typed canonical replay
qualify the source packaging only. Model-facing debt is recorded before publication; live sampling
and training admission remain separate gates.
