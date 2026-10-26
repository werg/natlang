---
description: Curate a field-observation batch while keeping raw provenance, candidate identity, event relations and public disclosure separate.
args: { collection: CurationCollection; request: CurationRequest; checkpoint: CurationCheckpoint; curator: CuratorIdentity }
returns: CurationProposal
---
Treat every submission as an immutable provenance envelope. Preserve original wording, timestamp string, declared timezone, method, location reference, media hashes, observer identity and evidence versions. Do not replace an original value with a corrected value.

Bind normalized = `normalizeSubmissionEnvelope(collection, request, checkpoint)` and bind the normalized observations plus unresolved metadata. A timezone correction requires a sourced offset and produces a new interpreted time; it never edits the original timestamp.

Bind interpretations = `resolveCandidateIdentities(collection, normalized, request)`. Retain every plausible concept when observations do not discriminate; a rare but sourced report remains eligible for expert verification. If evidence is unusual or implausible, request the specific image, field mark, or method detail that would distinguish the candidates. Never erase the observation as an outlier.

Bind links = `linkSubmissionEvents(collection, normalized, checkpoint, request)`. Distinguish same-event media attachments from independent observations. Use exact media hashes and scoped observer/time/location comparisons; nearby records or a shared species name are not sufficient to merge separate events. Preserve unresolved relation when location or timestamp precision is unavailable.

For decisive unknowns, Bind questions = `formFieldQuestion(collection, interpretations, request)`. Ask only for information an observer can safely provide from the already completed observation. Do not lead with a preferred label or ask the observer to return to an unsafe location.

Bind publicProjection = `projectPublicRecords(collection, normalized, interpretations, links, checkpoint, request.corrections)`. Apply the collection's disclosure rule to public location precision while preserving the original restricted location reference. Append expert corrections as new interpretation versions; invalidate only derivative IDs bound to the superseded interpretation and keep historical analysis traceable.

Return a `CurationProposal` for `curator.id` with unresolved alternatives, duplicate graph, safe questions, privacy findings, corrections and a type-compatible `resumeState` containing accepted provenance envelopes. Ready means reviewable for release; it does not mean public publication occurred.
