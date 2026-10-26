---
description: Reconcile archival citation lineages so derivative accounts do not count as independent witnesses.
args:{request:CommissionRequest;sources:ArchiveSource[];observations:SearchObservation[]}
returns:{sourceId:string;rootSourceIds:string[];questionIds:string[];independent:boolean;missingAncestors:string[];reason:string;}[]
---
Follow only linked identifiers explicitly present in hit observations and source records. Treat absent linked records as missing ancestors with unknown contents. For each account, trace the supplied citation chain to its earliest observed root. Two accounts sharing a single root are derivative support for that root, not independent corroboration. Preserve separate roots only when the source revisions and custody histories establish distinct provenance. Return each account's root source IDs, exact question links, independence assessment, missing ancestors and reason; never infer ancestry from similar wording alone.
