---
args: {"request": "ReaderEditionRequest", "edition": "ReaderEditionState | null"}
returns: "ReaderEditionReview"
---
1. Bind prior to edition or an empty revision-zero edition. Bind units = recoverOperativeMeaning(request.authoritativeSources, request.sourceChanges, prior.units). Preserve every consequential condition, exception and uncertainty before rewriting.
2. Bind tasks = modelActualReaderDecisions(request.projects, units, request.fidelityRules). Reorganize information around those decisions without omitting a required unit or stereotyping the reader's ability.
3. Bind paragraphs = writeTaskCentredDraft(tasks, units, prior.paragraphs). Keep technical terms where a plain synonym would change meaning; label examples as illustrative and retain the exact rule scope.
4. Bind probes = designConsequenceProbes(units, tasks). For each probe separately call answerFromAuthoritativeUnits(probe, units) and answerFromReaderDraft(probe, paragraphs), collecting their actual results in sourceAnswers and draftAnswers keyed by probe ID. Pass only the corresponding text to each reader. Bind differences = compareConsequentialAnswers(probes, sourceAnswers, draftAnswers, paragraphs).
5. Return material differences to the owning paragraph for source-grounded repair; preserve unresolved source ambiguity instead of forcing a cleaner rule. Rerun only affected probes after a revision, preserving every still-relevant exception/deadline boundary.
6. Bind release = checkReaderEditionBoundary(request, prior, units, paragraphs, differences). Return the new edition with its complete side-by-side source map, blocked paragraphs and invalidated reviews. ReleaseReady requires independent fidelity and actual headings/reading-order/link review for this exact edition; a shorter draft or better readability score is not that evidence.
