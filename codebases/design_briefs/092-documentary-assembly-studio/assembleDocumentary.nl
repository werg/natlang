---
description: Evaluate and freeze a documentary cut graph with source-context, rights, and duration controls.
args: { sourceRecord: DocumentaryProject; sourceEvidence: EvidenceItem[]; assets:MediaAsset[]; assembly:AssemblyState; corrections:EvidenceItem[]; checkpoint:DocumentaryCheckpoint }
returns: AssemblyRelease
---
Use the DocumentaryProject top-level interviewIds, archiveAssetIds, editorialBrief, editorId and targetDurationSeconds as typed arguments to argument mapping, cut review, clearance and freeze. Match them to the actual cuts and sourceEvidence IDs; retain unmatched project sources as omissions and block any cut whose argument exceeds editorialBrief. Preserve editorId as the reviewer binding. Keep the supplied correction artifacts attached to the exact source and timeline version.

Bind indexed = `indexSourceRanges(assets, assembly.cuts)` to bind each selected range to its timecode, source revision, speaker, qualifier, context and permission. Bind argument = `mapDocumentaryArgument(assembly, indexed, sourceRecord, sourceEvidence)` to relate each beat to approved claim identifiers and to distinguish thematic juxtaposition from causal statement.

Bind findings = `inspectCutImplications(assembly, indexed, argument, sourceRecord, sourceEvidence)`. Evaluate individual quotation fidelity and the proposition created by every transition. Restore required qualifier/context or block an implication that lacks independent evidence. Compute total frames from exact in/out ranges at the assembly frame rate; if it differs from targetFrames, produce a duration finding and keep release blocked.

Bind clearance = `reviewDocumentaryClearance(assets, assembly, indexed, corrections, findings, sourceRecord, sourceEvidence)`. An expired or channel-mismatched permission blocks the affected range; a speaker correction opens fact-check work and remains visible in the source claim history. If all blockers are resolved, return `freezeDocumentaryAssembly(assembly, findings, clearance, checkpoint, sourceRecord, sourceEvidence)` to bind the EDL, script, captions, descriptions and manifest to one assembly revision digest. On unknown prior dispatch receipt, demand readback before any resend.
