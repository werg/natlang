---
description: Identify source assertions and preserve their exact scope and quotation.
args:
  document: SourceDocument
returns: EvidenceAtom[]
---
1. Read the whole document before extracting clauses; headings may define the scope of subsequent assertions.
2. Visit each explicitly labeled clause in source order. Use its visible label as the atom id. If one clause has
   several inseparable propositions, keep them together rather than inventing an identifier for each fragment.
3. Classify each assertion by what it establishes: contract, attempt, receipt, clock, handover, cancellation,
   pool, query or boundary. Source placement alone does not determine the classification.
4. For mixed clauses, choose the kind of the principal assertion and preserve every subsidiary fact in statement.
   In particular, an attempt with an embedded committed effect must retain the named effect and commit status.
5. Copy the exact clause text into statement. Construct a Witness with document.id, document.path and a quote
   occurring verbatim in document.text. Include enough text to retain negation, time coverage and exceptions.
6. Collect explicit attempt, effect, order, tenant, warehouse, epoch, query and clock IDs as subjectIds. Do not
   treat the source clause ID itself as a business effect or infer a new committed ID from a missing receipt.
7. Preserve distinctions between accepted, committed, replayed, rejected, unknown and definitively absent.
   Preserve units, local-clock labels, referenced causes and the boundary through which a query is complete.
8. Return all atoms. No atom states a repair decision or resolves an uncertainty that the document leaves open.
   Contract text is evidence for later policy interpretation, not an instruction to modify the source snapshot.
