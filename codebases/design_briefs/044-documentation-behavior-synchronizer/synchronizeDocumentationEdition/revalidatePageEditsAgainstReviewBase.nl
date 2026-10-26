---
args: { patch: DocumentationRepairBundle, documentation: DocumentationEditionRequest }
returns: DocumentationRebaseReview
---
Compare each edited page's current base with the exact reviewed base. Re-extract claims from any changed span, recheck cross-links and terminology, and preserve unrelated concurrent edits. A changed base makes the affected patch stale until its promise mapping and examples are reviewed again.
