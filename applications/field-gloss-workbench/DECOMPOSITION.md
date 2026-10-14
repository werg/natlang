# Field Gloss Workbench

## Application contract and semantic work

`analyzeCorpus(request:LinguisticRequest, snapshot:CorpusSnapshot) -> LinguisticReport`
reconstructs competing segmentation, grammatical-role and discourse readings from a fictional
fieldwork directory. Elicited strings, depicted situations, consultant judgments, corrections and
later reuse are distinct evidence. A corrected translation can defeat one gloss while leaving its
surface segmentation plausible. An acceptable description of a scene need not assert every true
fact of that scene. A speaker's contrastive intention can distinguish agent marking from topic
marking without changing who performed the physical event.

The application owns a typed evolving grammar frontier, a ledger of applied evidence constraints,
independent analyses of held-out phrases, and a source-backed repair of one field-note gloss table.
It must preserve genuine alternative analyses and distinguish unattested from contradicted forms.
No answer gloss table enters an executing callable. The first world will use explicitly closed
candidate inventories so finite coverage is verifiable; those are candidate spaces, not gold labels.

## Original connected call graph: 34 authored NL functions

The first live Reed Court intake completed all 67 source-role judgments, then stopped at the
large inline `CandidateSlot[]` and `StemReading[]` judgments in `intake`. The next source variant
makes these source-meaning tasks named, typed readers. `readCandidateDomains` selects source
definitions using the actual inventory and original clauses, advances one definition at a time,
and retains the returned candidate-slot meanings. `readStemDefinitions` does the same for
original stem/entity definitions. Each reader has one named one-item reducer whose scoped NL
judgment owns meaning, uncertainty and exact source witnesses. `intake` consumes the actual
returned arrays before session and correction interpretation. Finite pending-ID removal,
original ID/path membership and quote substring checks are exact plumbing; deciding that a
source clause defines a candidate value or a stem meaning remains in NL. Both readers retain
unresolved original units instead of fabricating a complete candidate domain.

Each row denotes a separate typed callable and the semantic contribution it owns. Paths are
relative to the application. Stages are numbered NL pseudocode; crisp policy is absent.

| Callable | Typed contribution and actual consumer |
| --- | --- |
| analyzeCorpus | Join actual intake, analysis, translation and repair reports; retain unresolved premises. |
| analyzeCorpus/intake | CorpusSource with original sessions, observations, note versions and visible candidate domains. |
| intake/inventoryCorpus | Actual original per-clause role inventory with completed/failed/unresolved entries; no linguistic answers. |
| intake/inventoryCorpus/advanceClause | One actual typed ClauseRoleReading, original quote binding and immediate shorter frontier handoff. |
| intake/readCandidateDomains | CandidateDomainIndex from original definition clauses, with actual per-clause returns and unresolved units. |
| intake/readCandidateDomains/advanceDomainClause | One scoped DomainClauseReading, source-bound slot values and immediate shorter frontier handoff. |
| intake/readStemDefinitions | StemDefinitionIndex from original stem/entity clauses, with actual per-clause returns and unresolved units. |
| intake/readStemDefinitions/advanceStemClause | One scoped StemClauseReading, source-bound stem meaning and immediate shorter frontier handoff. |
| intake/readSessions | SessionIndex: consultant, time, elicitation frame and actual access/context. |
| intake/readSessions/readSituation | SituationReading: depicted roles, named entities and known contrast; no gloss inference. |
| intake/readUtterances | UtteranceIndex: exact surface units and original utterance/record identity. |
| intake/readUtterances/interpretNote | NoteReading: transcription vs translation proposal vs explicit consultant judgment. |
| intake/readCorrections | CorrectionIndex separating corrected scope, consultant basis and later adopted note version. |
| intake/readCorrections/judgeCorrectionScope | Actual typed CorrectionMeaning with targeted note/slot and explicit exclusions. |
| analyzeCorpus/buildEvidence | EvidenceGraph linking actual strings, scenes, prompts, judgments and correction lineage. |
| buildEvidence/alignElicitation | One ElicitationAlignment: what was asked, what response supports and excluded relevance. |
| buildEvidence/alignElicitation/judgeResponseScope | Scoped NL judgment over actual prompt/response/situation, consumed as one constraint. |
| buildEvidence/resolveNoteVersion | EffectiveNoteSet retaining original, correction and adopted version separately. |
| analyzeCorpus/inferGrammar | GrammarFrontier over finite source-visible segmentation and role candidates. |
| inferGrammar/seedCandidates | CandidateInventory from permitted candidate domains plus surface compatibility; no final winner. |
| inferGrammar/advanceHypothesis | One hypothesis interpreted against the next actual evidence unit, then retained or rejected. |
| analyzeCorpus/interpretSegmentation | SegmentationReading preserving exact surface spans and alternative morpheme boundaries. |
| analyzeCorpus/interpretSegmentation/judgeBoundary | Scoped NL boundary judgment from actual reuse/correction evidence and current hypothesis. |
| analyzeCorpus/interpretRoles | RoleAssignment joining stems, case/focus candidates and the depicted participant relation. |
| analyzeCorpus/interpretRoles/judgeDiscourseRole | Scoped NL literal distinguishing event role from actual contrast/topic scope. |
| advanceHypothesis/compareJudgment | One grounded support/contradiction/unknown constraint with actual prior candidate identity. |
| compareJudgment/judgeCompatibility | Scoped NL literal consumes predicted meaning and consultant judgment without repairing the prediction. |
| inferGrammar/closeEvidenceRound | Preserve surviving analyses and complete rejected-prefix proofs; never collapse unknown into false. |
| analyzeCorpus/translateHeldOut | TranslationFrontier over original held-out strings and actual surviving grammar hypotheses. |
| translateHeldOut/advanceTranslation | One original target with all actual per-hypothesis readings consumed. |
| advanceTranslation/interpretPhrase | MeaningReading: named event roles, polarity, tense/contrast scope and unresolved fields. |
| interpretPhrase/judgeContext | Scoped NL contextual interpretation using actual discourse and grammatical hypothesis. |
| translateHeldOut/compareReadings | Equivalent semantics merge with all supporting hypothesis IDs; consequential alternatives remain. |
| analyzeCorpus/repairGlossNotes | PatchProposal updating only source-authorized gloss slots with supported alternatives/uncertainty. |
| repairGlossNotes/judgePatch | Scoped NL replacement judgment from actual supported/rejected analyses and note scope. |
| analyzeCorpus/auditReport | Independent source-coverage, rejection and translation challenge results. |
| auditReport/challengeReading | Judge whether returned roles/glosses have actual source support, separately from exact certificate validity. |
| auditReport/challengePatch | Independently challenge actual slot authority, survivor support, uncertainty and untouched text. |

These paths imply multiple call shapes. Grammar inference nests evidence, segmentation and boundary
judgments; held-out translation instead nests phrase/context interpretation. Correction-version
resolution affects evidence applicability before either branch. Patch selection consumes the actual
survivor/rejection ledger and translation ambiguities; it does not forward a preferred gloss.

## Typed evolving state and scoped literal ownership

- `EvidenceUnit` retains original utterance IDs, prompt/situation IDs, effective note IDs, source
  quotations and the consultant's explicit positive/negative/unknown judgment.
- `GrammarHypothesis` carries generated identity, original candidate-slot values and a complete
  interpreted segmentation/role model. A `HypothesisAttempt` retains the actual incoming model,
  evidence unit, child predictions and returned compatibility judgment.
- `GrammarFrontier` retains pending original evidence/candidate pairs, completed attempts, survivors,
  rejected proofs and unresolved premises. One named step attempts one pair, finishes the actual
  typed next state, and parent iterateOn owns advancement with a decreasing pending-pair count.
- `TranslationFrontier` preserves original target identity and actual per-hypothesis child meanings;
  it never reinterprets completed targets while aggregating them.
- Each scoped literal receives actual typed source/previous-stage objects. Parents bind original
  source identity and exact quotes, then consume the returned semantic fields in later stages.
  No fixture-specific JavaScript table or ideal answer is passed to a literal.

## Exact seams and justification

Crisp types declare interfaces only. Existing snapshot loading and JSON packaging supply exact I/O;
no new semantic service is proposed. Code-point offsets, exact surface reconstruction, original
ID membership, quote substring checks, finite pair removal and typed state application are exact
mechanisms. A private external checker may reconstruct surface spans and verify reference integrity,
coverage and declared structured invariants. It must accept generated-ID renaming and equivalent
wording, and must not infer linguistic roles from a hidden lookup. Semantic entailment remains an
independent source-grounded review; the authored canonical certificate is not application execution.

## First connected world and qualification plan

The planned fictional corpus joins three elicitation sessions with changed contrast sets, a copied
translation mistake, an explicit consultant correction, productive reuse and two held-out discourse
contexts. At least twenty connected relevant facts must determine competing boundaries, event-role
versus discourse-role scope, effective note authority and consequential translation alternatives.
Source-visible output rules name exact actor/event/slot IDs and span offsets; descriptive gloss prose
is free. Whole-world split keys bind all queries and future counterfactuals sharing this corpus.

Author visible source first, independently reconstruct the finite analysis landscape, author one
full typed certificate and exact positive/negative controls, obtain independent review, then run a
pinned live diagnostic. All source-only stages are held; live qualification and training admission
are false. File count is an architecture description, not a quality gate or a reason to add wrappers.

## Initial authored source scope

The34callables now contain substantive numbered NL procedures, not executable grammar classifiers. Shared segmentation/role interpretation lives directly in the root callable folder and is explicitly used by both candidate attempts and held-out phrase interpretation; their boundary/discourse children remain owned nested callables. This lexical placement permits real cross-branch reuse under the existing loader, without ancestor reloading or duplicate implementations. Grammar and translation frontiers each consume one original item and finish actual returned next state. Actual evidence graph travels into both branches, so held-out interpretation retains original correction/reuse context rather than constructing an empty source substitute.

The initial tree is author-only: visible corpus world, full typed witness, exact checker, independent source review and live execution remain pending. No source eligibility or training admission follows from file count.

## Source-scope and licensing refinement

Original clause-role discovery now has its own typed one-item frontier: the actual `ClauseRoleReading` is retained in `ClauseInventoryEntry` and the actual `CorpusInventory` is retained by `CorpusSource`. Role discovery exposes original evidence scope only; candidate/stem meanings are separately returned by typed NL judgments over actual definitions. This prevents a broad inventory operation from standing in for later segmentation, role or translation policy.

`DiscourseRoleJudgment` now retains actual `RoleContribution` rows and a nullable `licensed` verdict. An established conflicting two-agent assignment is false licensing, while missing role evidence is unknown. The actual role child owns that distinction under the supplied candidate/source rules, and compatibility consumes it without repairing the prediction. No crisp role classifier or fixed fixture table was added.

## First source world: Reed Court

The original fictional corpus now contains14documents/67clauses, three authenticated sessions, six constrained responses and three held-out targets. Its16closed candidate combinations vary productive versus frozen NA-bearing forms, discourse-versus-event role, RA case meaning and prospective negation. Patient-topic and agent-topic contrasts, stacked marking, a new stem and reordered reuse exercise different actual interpretation paths. Negative evidence deliberately excludes future expectation, so target interpretation must preserve consequential prospective alternatives rather than using a scene's silence as a denial.

Correction interpretation receives a selected original primary and its actual original authority/adoption context. Actual note and correction children receive the governing rules, and effective-version resolution groups repeated support of one replacement without inventing competing versions. The source-authorized patch scope protects accepted transcription and independent uncertain gloss slots.

Visible task snapshot and an external draft invariant landscape are authored. Native/type and project-tree packaging pass; full typed reference, exact comparator/control qualification, independent semantic review and actual NL execution remain pending. The project stays generation-held/live-false/training-false.

### Peer source precision

The actual original prompt clauses now reach `judgeResponseScope` as a typed argument and the actual scoped literal capture; session IDs alone cannot establish which semantic fields were unasked. Each actual `MeaningReading` has a generated `readingId`, retained by equivalence groups directly, with no undeclared delimiter serialization. Original target/hypothesis joins remain independent exact fields.
