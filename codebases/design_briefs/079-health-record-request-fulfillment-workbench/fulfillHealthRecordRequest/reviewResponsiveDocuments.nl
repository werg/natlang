---
description: Separate responsive content from third-party information and route only affected spans for review.
args: { dossier: RecordsDossier; change: FulfillmentChange; authorization: ReleaseAuthorization; selection: RecordSelection }
returns: DocumentReview[]
---
For each responsive document, inspect the requested episode and categories. Mark relevant spans and third-party spans separately. Route only the affected spans for targeted qualified review; do not automatically release an entire document containing another person's information. Preserve unrelated responsive content when it is separable and authorized.
