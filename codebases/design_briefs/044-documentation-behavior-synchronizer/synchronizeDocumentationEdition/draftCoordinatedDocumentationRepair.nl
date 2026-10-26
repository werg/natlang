---
args: { mismatches: DocumentationDiscrepancy[], documentation: DocumentationEditionRequest }
returns: DocumentationRepairBundle
---
Draft only the changes authorized by the resolved repair direction. Keep unaffected claims and page spans intact. Coordinate page, example, test, and release-note edits against the same exact base. If an example contains a secret, block export and request credential rotation outside this documentation patch. If a deprecated service is demonstrated, mark its environment and attach the reviewed replacement.
Bind secretCheck = scanForSecretBearingExamples(documentation.exampleSources). Remove any flagged value from the candidate export and preserve the secret category and source span only.
