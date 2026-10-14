---
description: Assemble the certificate from actual source, arbitration, performance and constraint contributions.
args:
  request: EditionRequest
  archive: Archive
  families: FamilyIndex
  readings: ReadingIndex
  editions: EditionSet
  checked: CheckedEdition[]
  ambiguities: AmbiguityFinding[]
returns: EditionCertificate
---
1. Check request.caseId against the original archive. Verify source coverage and original quote/ID
   membership across all returned families, bundles, hypotheses, performed events and findings.
2. Confirm every checked edition refers to an actual assembled hypothesis and every performed event
   refers to that hypothesis's actual reading option for its written event and occurrence section.
3. Check state continuity within each voice/occurrence: prior afterAccidentals equals next before,
   and exact elapsed ticks agree with actual onsets/durations. At boundaries consume the source reset
   convention and ensure a reprise did not silently choose a different source-level correction target.
4. Retain actual rejected extensions with priorOptionIds/priorAssignments and incoming option/assignment.
   Verify every referenced option exists in actual bundles and the carried assignments match those
   actual option objects. The prior option event IDs must cover exactly the bundleOrder prefix
   before the incoming event, one option per location; an exclusive-claim rejection needs a real
   conflicting shared claim target. Retain every actually rejected inconsistent extension.
   Retain false/null ensemble findings with their original explanations.
   A report of all supported editions cannot silently omit one because of generated identifier order.
5. Copy readings.bundleOrder into the certificate as the exact reducer chronology.
   Assemble EditionCertificate from actual families, reading bundles, checked editions, rejected
   hypotheses, ambiguity findings and original coverage. Accumulate source and stage unresolved issues;
   do not rewrite the score or infer a new correction while auditing the certificate.
6. Return the typed certificate. Exact provenance, arithmetic and finite-domain checks can certify
   integrity; whether the NL interpretations follow the evidence remains independently reviewable.
