---
args: {"left": "QualifiedProposition", "right": "QualifiedProposition", "pages": "PublicationPage[]"}
returns: "PropositionComparison"
---
First align subject, audience, jurisdiction, effective interval, modality and source edition. If scopes cannot overlap, return coexists; if alignment is unknown, return unresolved. Within a supported overlap ask whether both qualified propositions can be true. Preserve negation and exceptions in that comparison. Distinguish explicit edition lag from incompatibility within one edition. Lexical difference and duplicate wording are neither proof of contradiction nor independent support.
