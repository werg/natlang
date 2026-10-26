---
description: Observe customer problem episodes, count independent incident events, maintain stable mechanism concepts, and report only comparable population trends.
args: { question: ObservatoryQuestion, observations: SignalObservationSet }
returns: ObservatoryResult
uses: [customerObservatory/assembleRequest, customerObservatory/assembleDossier, customerObservatory/frameObservatoryQuestion, customerObservatory/extractCustomerSituations, customerObservatory/versionCustomerIdentity, customerObservatory/deduplicateIncidentEvents, customerObservatory/versionProblemConcepts, customerObservatory/revokeAttributionDerivatives, customerObservatory/compareTrendCoverage, customerObservatory/bindTrendAggregates, customerObservatory/commissionDiscriminatingInvestigation]
---
1. Bind request = assembleRequest(question) and dossier = assembleDossier(observations). Preserve the question's original time window and population; do not infer omitted source categories or receipts.
2. Bind frame = frameObservatoryQuestion(request, dossier.studies). Freeze population definition, observation window, question and study version.
3. Bind episodes = extractCustomerSituations(request, dossier.rawEpisodes, dossier.evidencePages). Preserve raw source IDs and separate requested features from experienced problems.
4. Bind identityLedger = versionCustomerIdentity(episodes, dossier.identityMapVersions) before any customer counts. Migration joins require a versioned identity map; unresolved IDs block affected denominators.
5. Bind consentPlan = revokeAttributionDerivatives(episodes, dossier.consentChanges) before selecting public quotations. Withdrawn attribution requires removal and recomputation proposals; non-attributable incident evidence remains with its provenance.
6. Bind incidents = deduplicateIncidentEvents(episodes, identityLedger, dossier.outageReceipts). Collapse copied messages and outage tickets as one event volume, while retaining actual repeat failures on separate occasions.
7. Bind problemMap = versionProblemConcepts(incidents, dossier.conceptRenameEvents, dossier.evidencePages). Keep concept IDs stable across renames and inspect counterexamples before moving episodes.
8. Bind trendCoverage = compareTrendCoverage(request, dossier.samplingFrames, identityLedger, dossier.trendReceipts). A disappearing channel, incomplete sample or unresolved migration breaks comparability and forbids improvement claims.
9. Bind trends = bindTrendAggregates(request, problemMap, trendCoverage, dossier.trendReceipts) only when exact receipts exist and coverage is comparable; otherwise preserve the unavailable trend state without inventing a count.
10. Bind investigations = commissionDiscriminatingInvestigation(request, problemMap, trends, dossier.evidencePages). Select mechanism uncertainty that could change a product decision and cite both evidence directions.
11. Return the assembled request, frame, episode register, identity ledger, consent plan, deduplicated incidents, stable concept map, trend disposition and investigation queue as ObservatoryResult. Counts and external deletion effects require actual typed service receipts.
