---
description: Classify collection access and supplied observations for one commission question scope.
args:{request:CommissionRequest;collection:ArchiveSource[];question:string;prior:SearchObservation[];observations:SearchObservation[]}
returns:{collectionId:string;question:string;access:'open'|'closed'|'restricted'|'missing';queryVariants:string[];matchingObservationIds:string[];coverage:'observed-hit'|'observed-no-hit'|'closed'|'restricted'|'not-searched'|'partial';reason:string}
---
Bind the source set to exactly one requested collection. If none exists, mark access missing and do not infer holdings. For this question, enumerate the meaningful catalog terms, name variants and date/subject limits needed to cover the commission. Match prior and newly supplied observations only when they name this collection and their query fits this question's scope. Distinguish a fully covered no-hit from a partial search with unsearched variants. A hit requires supplied source IDs; a catalog reference alone does not establish the material's contents. Return exact observation IDs and a reason for open, closed, restricted, missing or partial coverage.
