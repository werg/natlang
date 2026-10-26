---
args: { request: DisclosureReviewRequest, receipts: RedactionReceipt[], renders: RenderReceipt[] }
returns: ReleaseArtifact[]
---
For each verified sanitizer receipt, inspect only render receipts for the same documentId and outputHash. Require the document to be in request.documentIds, a nonempty renderedHash and extractedTextHash, an actual visualReviewRef and no residualFindings. A stale, conflicting or incomplete render produces no ReleaseArtifact for that document and must be rerendered; never use the hash of a different output to fill a missing field.

For each uniquely verified pair return ReleaseArtifact { documentId: receipt.documentId, originalHash: receipt.inputHash, sanitizedHash: receipt.outputHash, renderedHash: render.renderedHash, extractedTextHash: render.extractedTextHash, redactionMapRef: receipt.redactionMapRef, recipientId: request.recipientId, purpose: request.purpose }. The recipient and purpose bind the proposed artifact for the next approval check; these fields do not assert consent. Return the array of those actual pairs, including an empty array when none verifies.
