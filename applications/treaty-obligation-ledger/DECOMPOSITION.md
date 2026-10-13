# Treaty Obligation Ledger — decomposition before implementation

Status: proposed third Astra application; fictional source law only. No runnable, live or training claim.

## Task and distinct structure

A fictional river federation has instruments issued by a council, delegated port offices and a temporary emergency board. Accession, reservations, amendments, sunset clauses and event-triggered duties interact. The application receives an immutable nested instrument archive plus a dated event docket. It returns an obligation ledger over requested time slices, with proof/defeat provenance and an explicit delta after a late document is discovered.

The hard question is which authority can change which obligation, for whom and during which interval. Publication, adoption, effective date and event time remain separate. An issuer may have power to create a notice duty without power to exempt a payment duty. A reservation may limit the consenting party's undertaking while leaving other parties' duties unchanged. An exception whose prerequisite is unknown remains unresolved. Two offices citing each other cannot manufacture the original authority neither possesses.

This is a versioned rule and justification system. It does not choose a route, pick a physical world or schedule a repair. Natural-language functions interpret delegated competence, clause scope, exceptions and document relation; finite fixed-point and dependency reducers maintain their actual results. The first fixture uses a fully fictional institutional rulebook, so external legal knowledge has no role.

## First connected source world

The **Three Locks Compact** covers ports Alder, Birch and Cedar and two river reaches, Upper and Lower. The charter grants the council tariff authority and delegates only notice wording to port offices. A flood board temporarily controls passage restrictions in the Upper reach. The compact creates toll, flood-notice and safe-passage duties, while a rescue exception requires documented rescue status. Birch accedes with a reservation about its own notice translations, which cannot waive another port's notice to Birch. A council amendment changes toll rates prospectively. An office circular attempts a tariff exemption outside its delegation. A signed rescue dispatch arrives after an initial docket but concerns an earlier trip; a later amendment cannot retroactively change that earlier trip. A board order sunsets before the final trip. Reciprocal office letters purport to delegate tariff powers to each other and must remain unsupported.

The initial snapshot now has 37 connected clauses across charter, competence grants, accession, compact schedules, amendments, orders, office circulars, trip events and discovered evidence. Each clause will be linked to an actual authority, applicability, defeat, derivation or delta consumer; no filler records. The request compares three trip times and one evidence-snapshot update, with a proof certificate for every active, defeated and unresolved obligation.

## Intended callable graph

```text
reconcileLedger.nl                         # directory reducer / root
  indexArchive.nl                         # folder ownership and evidence joins
    interpretInstrument.nl                # document kind, dates, issuer, references
    interpretEvent.nl                     # dated action vs report/receipt dates
  buildAuthority.nl                       # competence DAG and temporal grants
    deriveDelegation.nl                   # finite grant frontier
      assessCompetence.nl                 # scoped NL literal: act within grant?
      detectUnsupportedCycle.nl           # unsupported reciprocal authority
  compileNorms.nl                         # typed rules from authoritative instruments
    classifyChange.nl                     # amendment, reservation, interpretation, repeal
      resolveTarget.nl                    # referent/schedule/clause scope join
    interpretConditions.nl                # conjunction, exceptions, evidence status
  evaluateSlices.nl                       # ordered event-time slices
    applySnapshot.nl                      # changed facts and directly invalidated proofs
      seedApplicability.nl                # actor/reach/time/role matching
        judgeScope.nl                     # scoped NL literal over one norm/event
      settleFrontier.nl                   # justification worklist / fixed point
        deriveConsequences.nl             # rule premises into a proof hyperedge
        resolveDefeat.nl                   # scoped NL literal: authorized scoped override
        propagateRetraction.nl            # remove only proofs relying on invalidated premise
      classifyObligation.nl               # active / defeated / unresolved / unsupported
  explainDelta.nl                         # minimal changed-premise explanations
    compareCertificates.nl                # stable business identity, old/new justification
    minimizeSupport.nl                    # finite source-subset deletion with proof check
  auditLedger.nl                          # independent source/proof/scope challenges
    replayJustification.nl                # DAG replay, typed premise references
      challengeAuthority.nl               # scoped NL literal: provenance reaches charter?
```

Target: 27 meaningful NL callables, approximately 600–850 total NL lines. The implementation will remove any stage without an independently consumed delta. Most bodies will contain numbered typed pseudocode, branches and explicit update steps; line count does not grant credit.

## Typed ownership

| Branch | Principal typed values | Owned transformation |
|---|---|---|
| Archive | `SourceInstrument`, `EffectiveInterval`, `EventAssertion`, `EvidenceStatus` | Separate instrument adoption/publication/effect dates and action/report dates; preserve exact witnesses. |
| Authority | `CompetenceGrant`, `AuthorityClaim`, `AuthorityProof`, `UnsupportedComponent` | Derive only authority supported by a root grant and its actor/act/reach/interval scope. |
| Norm compiler | `Norm`, `NormTarget`, `Premise`, `ChangeRelation`, `EvidenceQuestion` | Convert source clause meaning into typed actor, event, consequence and defeater structure; retain unresolved scope. |
| Slice evaluator | `SliceState`, `ObligationKey`, `Justification`, `DefeatDecision`, `PendingDelta` | Maintain candidate obligations and alternative proofs as applicable facts and authoritative rules change. |
| Delta explanation | `CertificateDelta`, `PremiseCut`, `StableSupport` | Explain which new evidence changes status and which apparently relevant later instrument does not. |
| Audit | `LedgerCertificate`, `AuthorityChallenge`, `AuditFinding` | Replay proof edges from source premises, inspect temporal/competence scope and expose unsupported cycles. |

An obligation key is a source-derived tuple of debtor, beneficiary, action kind, event ID and source-defined action. Generated proof/node IDs are arbitrary consistent references. Distinct proofs for one obligation are retained; retracting one proof does not retract the obligation if another remains.

## Algorithms in natural language

1. Reduce source folders in their declared roles, retaining exact clause IDs and quotation witnesses. Use actual source references to join event and instrument identities.
2. Seed authority only from charter roots. Iterate over unprocessed grants, deriving a grant only when its parent authority is supported and its scope stays within that parent. Keep unresolved candidates and strongly connected unsupported citation groups as evidence; a citation cycle is not a root grant.
3. For each instrument change, resolve the targeted clause and the actor to whom it applies. A scoped literal assesses the proposed legal act against a single supported competence grant. Consume its in-scope/out-of-scope/unknown result in the accepted authority edge.
4. Compile each supported clause into a typed norm. Separate an event prerequisite from a later proof of that prerequisite. Distinguish default absence under a declared closed docket from an open unknown.
5. Evaluate requested slices in order. Seed matching rule/event pairs through a scoped literal that captures actor roles, reach and effective interval; consume the result in candidate applicability.
6. Use `iterateOn` with a finite set of unprocessed rule/premise/status combinations. Each step derives new justification edges, applies one authorized scoped defeater, or retracts a proof whose premise was invalidated. Evaluate the positive consequence closure within each source-justified defeat stratum before propagating the next stratum. A cyclic negative-dependence component with no source rule that resolves it is reported unresolved and excluded from settled truth; it is never iterated until a convenient answer appears. Memoized derivations prevent circular self-support. Termination is exhaustion of source-defined combinations in the finite dependency strata, not a turn budget.
7. A scoped defeat literal judges one competing norm pair under their authority and target scopes. Priority is derived from the source's change rules, not from document recency alone. Unknown priority remains explicit.
8. Classify an obligation using all surviving justifications and defeaters. Keep active, defeated, unresolved and unsupported distinct. Preserve the exact reason a tariff circular failed even when a valid toll result exists.
9. Apply the discovered evidence as a new evidence snapshot, keeping historical event time fixed. Invalidate affected proof edges through dependency references; recompute their reachable closure. Compare obligation identities and proof statuses to the previous snapshot.
10. Minimize each claimed change explanation by removing one proposed changed premise at a time and replaying only the affected proof cone. The result is an inclusion-minimal sufficient support or a witnessed unresolved alternative; do not claim minimum cardinality unless exhaustively established.
11. Audit every retained obligation with an independent scoped authority challenge and exact premise-DAG replay. Keep contradictions and unresolved source scopes in the returned certificate.

## Literal consumption

- `assessCompetence`: one act + one grant; result gates an authority edge.
- `judgeScope`: one norm/event/slice; result gates applicability.
- `resolveDefeat`: one proposed override + authority/target context; result defeats one proof or preserves it.
- `challengeAuthority`: one final proof chain + root grants; result becomes an independent audit finding.

Each local function is created with typed `nl.with` captures and directly awaited on its changing input. The returned value must be read by the owning stage. Merely constructing a literal earns no contribution credit.

## Exact seams

- Existing folder and callable loader: immutable bytes, path containment and signature mechanism.
- Source date values and interval comparisons: exact arithmetic after NL determines which date controls the clause.
- Stable tuples, graph traversal, set membership and proof dependency references: exact mechanism; they do not decide authority or legal scope.
- External verifier: checks a submitted ledger against fixture-derived obligations, exact dates/amounts/statuses, source witness membership, proof acyclicity and alternative-support preservation. It never creates a norm or chooses which source should prevail.
- Canonical typed reference: external source-validity witness only. Prose explanations and equivalent proof encodings require independent semantic review.

No legal database, external law, crisp rule interpreter or policy engine supplies the answers. All institutional semantics are original fictional source facts.

## Qualification order

Connected fixture and independently derived oracle; then typed end-to-end source slice; then full declaration/callable validation, equivalent-positive and adversarial-negative verifier checks, independent review, shared source replay and live execution. Training admission and the roughly 48-sample model-facing qualification remain separate. Preserve all earlier valid simple cases.
