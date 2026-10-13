# Turn Window Adjudicator

## Task and source boundary

Adjudicate one closed fictional tabletop turn from nested rules, card texts, announcements, responses and the before-turn board snapshot. Return the resolved board, an ordered window trace, replacement/trigger provenance and explicit unresolved choices. Source text defines the entire game. Familiar words such as damage, shield or controller acquire only the meanings given here.

The new work is a semantic event interpreter, not a planner choosing a winning move. Submitted intents and choices are immutable inputs. Each rule application changes a typed event bundle or board delta that a later stage consumes. The final report is reconstructed from those actual deltas. Existing simpler game sources remain eligible.

Target organization: 26 meaningful NL callables, approximately 500–700 substantive lines, five scoped literal sites. These are design estimates, not qualification. No policy is implemented in TypeScript. Type declarations and external exact replay checks are the only proposed crisp files.

## Core typed data

- `SourceClause`: original clause identity, source path, text and exact witness origin.
- `RuleReading`: ability identity, event class, timing window, scope, replacement/trigger distinction, controller basis, source-defined use tokens and unresolved roles.
- `BoardSnapshot`: object instance IDs, object revisions, zone, controller, counters, attachments and resources. Object identity survives a controller change but follows the source's zone-change rule.
- `Announcement`: submitted intent, announced object/target IDs, captured controller and quantities, captured state revision and supplied choices.
- `EventBundle`: one resolving cause with one or more typed effect parts; each part carries subject identity, quantity, source, application history and source witnesses.
- `ReplacementDecision`: applicability true/false/unknown, semantic rewritten parts, consumed use-token IDs and supporting source clauses.
- `WindowState`: announced stack, pending interrupts, current board, deferred trigger tokens, finite `remainingWorkSlotIds`, event ledger and unresolved contributions. Every field has one owning reducer.
- `BatchDelta`: mutually simultaneous proposed resource/zone changes computed from one before-batch snapshot, plus emitted trigger proposals. A sibling event cannot read another sibling's unapplied change.
- `TriggerDecision`: qualifying resolved event IDs, source controller basis, eligibility, quantity, target binding, consumed one-use token and witnesses.
- `TurnCertificate`: final board, ordered windows, event rewrite histories, simultaneous batches, actual trigger decisions, source coverage and unresolved choices.

Generated trace IDs may be alpha-renamed. Original object, rule, event and intent IDs are exact. Event-part order is irrelevant within a simultaneous batch; window order and causal parent direction are binding. Explain source-defined identity changes in typed links, never in a naming convention inferred from an oracle.

## Callable graph and ownership

| Callable path | New NL judgment and owned typed delta | Consumer |
|---|---|---|
| `adjudicateTurn` | Compose source index, captured announcements, settled windows and audited certificate. | Root result |
| `adjudicateTurn/readTurn` | Reconcile source roles and closed-turn inventory. | Compiler and announcement capture |
| `readTurn/readRules` | Parse normative ability/timing clauses; retain missing roles. | `RuleIndex` |
| `readRules/interpretAbility` | Scoped literal distinguishes replacement, permission, cost and triggered consequence. | Actual `RuleReading` |
| `readTurn/readBoard` | Join physical object instances, attachments and resource owners. | Initial `BoardSnapshot` |
| `readTurn/readIntents` | Interpret announced effects and explicitly submitted responses/choices. | `IntentIndex` |
| `adjudicateTurn/compileWindows` | Derive precedence and capture/current-state rules from source timing. | `WindowProtocol` |
| `compileWindows/classifyTiming` | Resolve announcement, before-resolution, simultaneous-resolution and after-resolution roles. | `TimingBinding` |
| `compileWindows/checkWindowDependencies` | Identify inconsistent or cyclic precedence rather than inventing a priority. | Protocol conflicts |
| `adjudicateTurn/captureAnnouncements` | Own immutable announcement-time targets, quantities and controllers. | Window seed |
| `captureAnnouncements/bindReference` | Scoped literal distinguishes tracked object, current controller and captured controller references. | `ReferenceBinding` |
| `adjudicateTurn/settleTurn` | Own finite event/interrupt/trigger frontier. | Settled trace |
| `settleTurn/advanceWindow` | Choose the next source-defined ready window; honor supplied choices. | One `WindowState` transition |
| `advanceWindow/resolveInterrupts` | Validate and apply submitted response effects before the suspended event resumes. | Updated current board and spent response IDs |
| `advanceWindow/prepareBundle` | Materialize one effect bundle from actual announcement and resumed state. | Replacement frontier |
| `advanceWindow/rewriteBundle` | Own event-part rewrite state and per-part applied-rule history. | Final event bundle |
| `rewriteBundle/applyReplacement` | Scoped literal decides source applicability and produces typed rewritten parts. | Actual replacement delta |
| `rewriteBundle/resolveOrdering` | Decide whether source rules commute; use submitted source-legal choice when they do not. | Replacement order or unresolved requirement |
| `advanceWindow/resolveBatch` | Derive all effect outcomes from the same before-batch board. | Simultaneous `BatchDelta` |
| `resolveBatch/judgeOutcome` | Interpret damage/prevention/transfer/destruction conditions without observing sibling unapplied changes. | Per-part outcome |
| `advanceWindow/collectTriggers` | Select triggered consequences from resolved events rather than announced intent. | Typed trigger queue |
| `collectTriggers/interpretTrigger` | Scoped literal joins event history, last-known source identity and controller basis. | Actual `TriggerDecision` |
| `resolveBatch/reconcileDeltas` | Decide whether same-object proposals accumulate, replace or conflict under the supplied simultaneous-change rule. | Source-justified merged batch or unresolved conflict |
| `adjudicateTurn/auditTurn` | Compare final state, event histories and source duties; expose missing semantic contributions. | Certificate |
| `auditTurn/replayIdentity` | Audit object identity and controller transitions across windows. | Identity findings |
| `auditTurn/challengeTiming` | Scoped literal tests a claimed trigger/replacement against its actual timing and snapshot. | Audit findings |

The paths in this table abbreviate nested callable directories after their owning parent. Implementation will preserve the actual edge graph; no global helper is callable by a sibling merely because it has a familiar name.

## Numbered NL algorithm

1. Read and classify each original source once. Keep rules, instance declarations and intent records distinct; a quoted alternative example is not an announced action.
2. Compile the source timing protocol and bind original object identities. Preserve unresolved timing or referent clauses before execution.
3. Capture announcement-owned fields in immutable `Announcement` values. Store which properties were captured and which remain live references.
4. From the closed source inventory, allocate one work slot for each submitted announcement, submitted response and source-authorized one-use trigger consequence. Initialize the active frontier with ready submitted work; inactive trigger slots already belong to the finite universe. Initialize each replacement slot from its source rule.
5. Advance the next permitted window. Resolve its submitted interrupts, then resume the suspended cause against the returned current board while retaining captured announcement fields.
6. Build its effect bundle. Repeatedly interpret applicable replacement rules, consuming each permitted rule token/history slot; each returned replacement actually changes the next bundle. If two rules do not commute and no legal source choice resolves them, keep that choice unresolved.
7. Evaluate all final parts against one immutable before-batch snapshot. Reconcile overlapping proposals using the source simultaneous-change rule, then produce one `BatchDelta`. Apply its already selected resource, attachment and zone changes together as one exact state-update instruction in `advanceWindow`.
8. Judge after-resolution triggers from the actual resolved event ledger. A fully prevented effect may differ from an announced effect or a partially resolved effect. Bind the recipient using the rule's captured/current/last-known controller basis.
9. Spend the exact tokens of emitted triggers and activate their already inventoried work slots in the frontier. Remove the completed window's work slot from remainingWorkSlotIds exactly once. Record negative trigger decisions too, so an absence has a source-based reason.
10. Repeat until the finite frontier is empty or a genuinely missing source choice prevents further progress. Audit each causal edge, snapshot read and conservation identity before returning a certificate.

## Structural termination

A replacement pass strictly reduces eligible unused replacement-history slots for its current finite bundle. Source rules can split one effect into a finite bundle, but cannot duplicate the same rule application on its own output lineage unless the source explicitly supplies another token.

Before execution, inventory a finite work-slot universe from submitted announcement IDs, submitted response IDs and all source-authorized one-use trigger consequence slots. The first world explicitly permits no new response submission during adjudication and allocates each trigger consequence once. A trigger may activate several ready slots only when those slots already exist in this source-derived universe; it never adds an identity to the universe.

The outer reducer owns remainingWorkSlotIds. One complete committed window removes its active slot exactly once; enqueuing another already inventoried slot does not change that set. Its measure is remainingWorkSlotIds.length, which therefore strictly decreases even when the ready queue grows. A finished predicate says that no active event or source-permitted submitted response remains; inactive trigger slots may remain when their conditions never occur. A missing source choice returns an unresolved terminal result rather than retrying an unchanged slot.

If a future source allows repeat triggers, newly created trigger capacity or newly submitted responses, its source-derived closure and termination proof must be revised before that source is accepted by this application. There is no arbitrary turn count or execution budget. The exact universe tracks permitted work identities; NL functions still decide applicability, order, replacement and trigger meaning.

## First connected world: Glass Garden turn

The fixture will use six physical pieces, two players, one announced split effect, two submitted interrupts and four one-use triggered abilities. The core interaction is a protection attachment that moves during a response, while one effect remembers its announced controller and another follows the same physical object. A replacement changes damage into counters; another reacts only to actual damage. Two sibling effect parts resolve simultaneously, so a destruction-derived trigger cannot change the sibling's pre-batch threshold. A returned object is a new instance under the explicit zone-change rule, making a last-known reference different from a live target search.

At least 30 connected clauses will establish every timing, identity, quantity, scope and negative case. A tempting announcement-only trigger, an exhausted token, a similarly named new object and an obsolete controller are explicit adversarial alternatives. The independent oracle will enumerate the submitted legal choices and show either a unique result or a small declared result set. No hidden priority rule, usual-game convention or unannounced player choice may make the result unique.

## Exact seams and qualification

- Existing Folder reads and loader preserve snapshot bytes and callable visibility.
- Parent-owned ID/path binding and exact quote substring checks validate literal provenance. Each nonempty semantic contribution needs nonempty supporting evidence.
- Basic arithmetic, set membership and application of already selected deltas are exact mechanism. They do not choose a replacement, resolve scope or determine a trigger.
- An external verifier checks supplied event histories, numeric deltas, token use, source IDs and the authored final-state invariants. It must accept harmless trace-ID renaming and sibling permutations.
- Independent semantic review checks timing entailment and alternative completeness. Canonical replay checks source/type validity; it is not execution evidence.
- The first live slice must invoke interpretation, reference capture, replacement, simultaneous resolution and trigger interpretation, with each actual child result consumed. Any pass-through stage found in that trace will be removed or given a real missing transformation.

Full source tree now authored:26 NL functions /643 total lines /435 body lines /162 numbered steps /43 declaration-only aliases. The root is `adjudicateTurn.nl`, returning TurnCertificate through actual child contributions. Native parser/type check passes. The connected Glass Garden fixture has10 documents/39clauses and an external draft invariant oracle. Full typed canonical reference, external exact verifier and independent semantic review remain pending; no live execution or training admission is claimed.

## Implementation composition audit

All effect selection and interpretation remain numbered NL pseudocode. The replacement reducer uses an actual scoped ReplacementDecision, preserving beforeParts/afterParts and source-defined consumed tokens. Simultaneous reconciliation joins typed proposals from one immutable before-board and derives departures after accumulation. Trigger interpretation consumes those departures and actual rewrite quantities, selecting captured/current/last-known recipients into typed consequence parts. Independent identity/timing audits inspect the actual returned trace. The only crisp operations requested in bodies are source reads, quote/domain checks, arithmetic, finite frontier bookkeeping and application of already selected deltas. No executable TS policy or semantic regular-expression parser is present.
