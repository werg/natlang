# Dialogue Perspective Ledger — decomposition before implementation

## Purpose and semantic problem

Read a connected fictional conversation from nested transcripts, forwarded quotations, private notes,
actual receipt records and independently verified physical events. Derive who endorsed which proposition,
what each participant had available at each event, which conditional commitments were made or revised,
and what a specific perspective can support at a requested time. Return a source-linked typed report.
The task is dialogue interpretation, not a legal obligation engine or an action planner.

The central distinctions are speaker versus reporter, quotation versus endorsement, original utterance
time versus forwarding time, factual truth versus participant belief, activation versus timely promised-consequent fulfillment, delivery versus adoption, private
revision versus a communicated correction, and a conditional promise versus an event that happened.
A report must preserve unsupported or genuinely ambiguous meanings. It cannot read another person's
private note into a recipient's state just because the application sees the complete evidence folder.

All semantic policy is numbered NL pseudocode. Parent stages consume actual typed child judgments and
apply returned deltas. The implementation target is 29 connected NL functions with roughly650–850
substantive lines. Source size is descriptive; no filler or template chain is credited as reasoning.

## Typed source and state

- `DialogueSource` retains original documents/clauses, authenticated turns, participant/channel scopes,
  chronology and receipt records. Exact framing preserves IDs/paths/text; interpreting content is NL.
- `QuoteTree` contains original span identity, nesting, attributed speaker, original temporal frame,
  reported/endorsed scope and unresolved attribution. A forwarded quote keeps its original speaker/day.
- `ActReading` retains focused speech units, illocution, referent bindings, per-proposition claim values,
  separate activation/consequent predicates, deadlines,
  audience and exact original support. Unknown irony or pronoun scope is explicit alternatives/unresolved.
- `ParticipantView` contains available event IDs, endorsed belief atoms, conditional commitments and
  revision lineage and actual adopted revision act IDs. Commitment effects retain separate issuer-owner
  and adopting-audience scope. Every derived atom identifies its participant perspective and supporting availability.
- `PerspectiveFrontier` owns the finite ordered original episode IDs, current views, actual deltas,
  physical-event facts and accumulated trace. Each step consumes one original episode.
- `ProofGraph` separates original evidence from derived belief/commitment/query nodes. Directed edges
  identify availability, endorsement, condition, temporal anchoring and revision roles; no prose ID encoding.
- `PerspectiveReport` contains actual episode traces, participant snapshots, typed query answers,
  proof graphs, defended/defect/unresolved audits and original source coverage.

Final interfaces will use named fields for compound relations and explicit ordering where chronology
matters. Generated proof/episode IDs may alpha-rename consistently; original source actor/turn/clause IDs
remain exact. Free explanation wording requires independent semantic review, not byte equality.

## Callable folders and owned transformations

| Callable .nl path | Owned NL transformation | Returned typed delta consumed later |
| --- | --- | --- |
| analyzeConversation.nl | Reduce immutable source folder, run scope/view/query branches, return audited report | PerspectiveReport |
| analyzeConversation/intake.nl | Join original transcript, private, delivery and physical-event source roles | DialogueSource |
| analyzeConversation/intake/readChronology.nl | Distinguish original utterance, forwarding, delivery and event-time axes | TemporalRegistry |
| analyzeConversation/intake/readParticipation.nl | Interpret speaker, addressee, channel and explicitly changing group scope | ParticipantIndex |
| analyzeConversation/intake/readReceipts.nl | Distinguish dispatched, received, read, doubted and explicitly adopted evidence | ReceiptRegistry |
| analyzeConversation/scopeTurns.nl | Combine actual quote trees, reference bindings and speech-act meanings | ScopedConversation |
| analyzeConversation/scopeTurns/partitionQuotation.nl | Own the finite original-span frontier and consume actual typed frame judgments | QuoteTree |
| analyzeConversation/scopeTurns/partitionQuotation/advanceQuoteSpan.nl | Consume one original span, judge its frame and activate strictly smaller disjoint quoted spans | Smaller QuoteSpanFrontier |
| analyzeConversation/scopeTurns/partitionQuotation/advanceQuoteSpan/judgeQuotationFrame.nl | Scoped NL judgment of one quotation's original author/time/reporting layer | QuoteFrameDecision |
| analyzeConversation/scopeTurns/resolveReferences.nl | Join one frame's person/place/time expressions to actual perspective | ReferenceBindings |
| analyzeConversation/scopeTurns/resolveReferences/judgeDeixis.nl | Scoped NL literal binds one expression in its original quote frame | DeicticMeaning |
| analyzeConversation/scopeTurns/classifySpeechActs.nl | Separate act units and consume actual illocution results | ActReading[] |
| analyzeConversation/scopeTurns/classifySpeechActs/judgeIllocution.nl | Scoped NL literal distinguishes assertion, report, request, promise, correction, irony and unresolved reading | SpeechActMeaning |
| analyzeConversation/assembleEpisodes.nl | Join original acts, receipts, notes and physical facts into source-established order | EpisodePlan |
| analyzeConversation/assembleEpisodes/joinAvailability.nl | Judge which participant had which original content available at this episode | AvailabilityReading |
| analyzeConversation/evolveViews.nl | Initialize and reduce finite participant-view episode frontier | PerspectiveEvolution |
| analyzeConversation/evolveViews/advanceEpisode.nl | Own one episode, call semantic delta children, apply only actual returned effects | Smaller PerspectiveFrontier |
| analyzeConversation/evolveViews/advanceEpisode/judgeEndorsement.nl | Scoped NL literal separates hearing/reporting/doubting from belief endorsement | EndorsementDecision |
| analyzeConversation/evolveViews/advanceEpisode/updateBelief.nl | Derive actor-owned belief addition, defeat or unknown from that actual endorsement | BeliefDelta |
| analyzeConversation/evolveViews/advanceEpisode/updateCommitment.nl | Interpret conditional promise, beneficiary, due frame and actual condition reading | CommitmentDelta |
| analyzeConversation/evolveViews/advanceEpisode/updateCommitment/judgeCondition.nl | Scoped NL literal distinguishes actual fulfillment, issuer knowledge and listener belief | ConditionReading |
| analyzeConversation/evolveViews/advanceEpisode/resolveRetraction.nl | Judge scope/recipient of correction and defeat only the actually superseded reading | RevisionDelta |
| analyzeConversation/answerQuestions.nl | Resolve every source-visible query against its requested perspective/time | QueryAnswers |
| analyzeConversation/answerQuestions/resolveQueryFocus.nl | Interpret belief, statement, commitment or verified-fact question without collapsing them | QueryFocus |
| analyzeConversation/answerQuestions/joinProof.nl | Join actual episode/view lineage into minimal sufficient source-backed support | ProofGraph |
| analyzeConversation/answerQuestions/joinProof/challengePerspective.nl | Scoped NL challenge checks wrong-speaker, private-knowledge, temporal and quote-scope assumptions | PerspectiveChallenge |
| analyzeConversation/auditReport.nl | Inspect actual deltas, proofs and source completeness without replacing them with a reconstructed answer | PerspectiveReport |
| analyzeConversation/auditReport/checkNoKnowledgeLeak.nl | Audit actual participant availability and private-versus-shared propagation | LedgerFinding[] |
| analyzeConversation/auditReport/replayAnswer.nl | Independently replay one answer's declared source/perspective/condition path | AnswerAudit |

## Different control paths

1. Quotation parsing uses a typed QuoteSpanFrontier and named advanceQuoteSpan.iterateOn; no NL
   function calls itself or an ancestor. Each step consumes one original span and adds only strictly
   smaller disjoint quoted spans whose combined character mass is less than that consumed span.
   Pending character mass therefore decreases. A nested report has another attributed frame; it does
   not become an assertion by the outside reporter. Exact original substring/range checks validate
   the NL-chosen source spans without classifying their speaker or endorsement.
2. Per-expression reference judgments use that actual frame. Exact day arithmetic follows the returned
   original temporal anchor; it never guesses which speaker's tomorrow was intended.
3. Episode reduction calls different semantic children for assertion/adoption, conditional commitment
   and correction. Returned actor-owned effects change state; a generic unchanged forwarding stage is
   not an implementation of any of these policies.
4. Condition judgments separately consume verified facts, issuer-available knowledge and listener belief.
   A mistaken assertion may produce a belief while failing an actual promise condition.
5. Query proof construction follows actual supporting and defeating edges. Independent challenges
   inspect quote scope and availability; a convenient global fact cannot repair an invalid private view.

## First connected source world

Planned original `portico-packet` world: five participants, two nested forwarding layers, an original
"tomorrow" anchored before the forward, an authenticated intermediate archive quote, a before-noon conditional promise, an acknowledgment tied to a
specific brief version, an explicitly mistaken collection claim, actual later collection, recipient doubt
and explicit adoption, a private correction reaching one recipient late, an ironic remark with signed
context, and an unresolvable pronoun retained as ambiguity. Every fact participates in a query or in
support/defeat/availability lineage. Original source rules explicitly define this fictional speech model;
no outside custom about promise, politeness, honesty or belief is silently imported.

The visible output contract names compound fields and query observables. A full typed canonical report
will be authored outside visible roots and independently reconstructed; it proves one witness's type and
source validity, not that the NL functions executed. The trusted checker will verify exact original IDs,
quotes, event order, declared receipt/perspective domains and authored typed query invariants. Illocution,
proposition paraphrase, belief entailment and proof minimality remain semantic review obligations.

## Crisp seams and justification

| Crisp piece | Reason exact mechanism is needed | Semantic decisions retained in NL |
| --- | --- | --- |
| Existing Folder/loader | Immutable original source bytes and callable-folder visibility | Source relevance, participant/quotation meaning |
| Type declarations | Validate named structured interfaces and nullable uncertainty | Act classification, reference/endorsement/condition judgments |
| Arithmetic and map/set updates | Apply already returned time offsets and actor-owned deltas exactly | Which frame, belief or commitment those operations represent |
| External verifier | Check original ID/quote domains, declared exact invariants and typed witness structure | No text classification, belief adoption or repair selection |
| Authored control fixture | Supply independent positives/corruptions and grounded expected query fields | Does not implement the application or replace live judgment |

No executable TypeScript semantic helper, regex act classifier or case-specific assertion table is planned.
No effect leaves the snapshot. No runtime call budget is a termination proof. Source/model-facing changes
update both repository authoring/integration skills and record approximately48-sample live debt until
measured. Independent source/oracle/checker review and full live trace qualification precede admission.

## Source handoff audit before canonical sealing

Per-proposition claims replace the single aggregate endorsement value: version acknowledgment and
collection correction contain opposite values. Typed SpeechUnit focus/support is passed to the actual
illocution child, so separate force units are not identical whole-node calls. ReceiptReading carries
actual verifiedFactIds into participant availability. Private intention has its own act kind and visible
proposition; it supplies no communicated cancellation. Adoption IDs are actual scoped act identities.
Retraction joins real beneficiary views and issuer-owned commitments, with separate owner/adopter fields
in the returned delta; later fact/deadline recomputation preserves terminal retraction.

## Canonical certificate and exact verifier boundary

One private authored certificate records14 original turns,18 scoped acts,22 episodes and16 query
answers. Its annotations are a source witness, not an executable semantic extraction algorithm or
an application trajectory. It is excluded from both visible fixture and application roots. The exact
checker validates original identities/quotes, clock order, declared availability bounds, actual typed
delta-to-history propagation, generated-reference integrity and declared query projections. It accepts
consistent generated-ID renaming, unordered source/quote/proof collections and free explanations.
It neither classifies utterances nor proves that a quoted source entails a returned speech meaning;
independent semantic and full live contribution review remain required. Knowledge observables test
whether the requested knowledge criterion qualifies, separately from the underlying proposition value.

Source-established cutoff/day-end closures are typed boundary episodes, separate from physical occurrences,
utterances and receipts. The episode NL stage derives their source role and clock from actual act conditions
and due fields, retaining original witnesses and granting no participant access by the boundary itself.

## Connected revision target variant

Countermand Bridge exercises a different audience/causal structure: one actual adopter does not
withdraw a two-beneficiary promise; a later original adoption completes the issuer-owned normative
revision while issuer knowledge remains absent. A fresh promise later fulfills without reviving the
terminal original, and actual receipt of a factual correction leaves belief unchanged without adoption.

Add one substantive NL child beneath resolveRetraction: selectRevisionTarget consumes actual act,
returned endorsement/adoption IDs, episode and original scoped/source objects. Its numbered policy
joins adopting act→original revision→original target, preserves target exclusions, and distinguishes
belief-correction, commitment-retraction, none and unresolved meaning. It returns RevisionSelection,
retained verbatim in RevisionDelta.selection. Parent uses those actual target/communication fields
before joining issuer-owned commitments and every beneficiary's actual access/adoption. No target
lookup service, hidden expected answer or crisp semantic classifier is added.

The episode reducer keeps actual child state deltas and finishes one updated typed frontier; parent
iterateOn owns the next source episode with explicit decreasing-count and empty predicates. Query
performed force is the requested force, including withdrawal, rather than any performed report.
The exact checker validates selected-ID ownership and new retraction provenance; it does not choose
semantic targets. First-world source bytes/certificate are retained in the frozen reviewed baseline;
current richer variant and second-world certificate await independent review/live qualification.
