---
description: Route question-specific archive searches and preserve exact coverage states.
args: { request: CommissionRequest; sources: ArchiveSource[]; questions:string[]; prior:SearchObservation[]; observations:SearchObservation[] }
returns:{observations:SearchObservation[];coverage:ArchiveSearchCoverage[]}
---
For every requested collection and bounded question, bind coverage = `classifyCollectionCoverage(request, sources, question, prior, observations)`. Retain only actual prior and supplied observation records, deduplicated by exact observation ID; never fabricate a hit or a no-hit. Return the full joined observation manifest and every question/collection coverage result for the dossier. A closed/restricted collection is never called searched and a partial set of query variants is not exhausted. Only supplied hits count as source evidence.
