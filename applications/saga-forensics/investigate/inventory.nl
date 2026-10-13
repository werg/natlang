---
description: Reduce the manifest-listed source documents to witnessed evidence atoms.
kind: directory-reducer
args:
  request: InvestigationRequest
returns: EvidenceIndex
---
1. Read manifest.json with folder.readText and parse it as SourceManifest. Check its caseId against request.caseId.
   Use documents as the exact finite inventory. Neither filesystem ordering nor a filename implies authority.
2. Initialize documents, atoms, coverage and ambiguities as empty collections. Partition the inventory by parent
   directory so the reducer visits each source collection while preserving every document's individual identity.
3. For each collection, read each listed relative path with folder.readText. Construct SourceDocument with the
   manifest id, path, roleHint and exact text. Retain that text unchanged for quote checking in audit.
4. Call interpretDocument(document) for each document and consume its EvidenceAtom[] into atoms.
   Record the document's id and all returned atom IDs in coverage. Preserve cross-file references as identifiers.
5. If an atom ID occurs twice with different statements, retain both witnessed assertions and add an ambiguity.
   Exact repeated observations still remain separate atoms; business-effect deduplication belongs to adjudicate.
6. Check that every non-heading source clause has a corresponding atom, possibly several when a clause states
   distinct facts. A compound statement must keep its exception or uncertainty attached to the assertion.
7. A document's roleHint is a navigation aid. A journal can include an explicit committed receipt and a receipt
   document can quote a mere acceptance. interpretDocument decides the assertion kind from the words.
8. Return the manifest, source documents, atoms, per-document coverage and source ambiguities. Keep contract
   clauses even when they do not mention the target orders: their scopes can protect shared resources.
