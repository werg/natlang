---
description: Read original nested bank records and preserve their biological source roles.
kind: directory-reducer
args:
  request: LaboratoryRequest
returns: BankSource
---
1. Own the BankSource source-inventory contract. Preserve the request/output convention as original
   source data for later stages. Read manifest.json through folder.readText and parse BankManifest.
   Require its caseId to equal
   request.caseId. Its finite listed documents define the source snapshot and visible output conventions.
2. Visit the actual parent-directory groups and read every listed document. Preserve id, path, role
   and text. Extract visibly labeled clauses as SourceClause with a nonempty verbatim Witness.
3. Keep original source identity attached to every clause; exact quote checks precede semantic use.
   Record coverage for all documents. Conflicting duplicate clause IDs remain unresolved.
4. Call readPedigree(clauses) and consume accession domains, parentage decisions and inheritance
   conventions. Call readTheories(clauses) for competing expression rules and joint prior tickets.
5. Call readAssays(clauses) for observation scope and assay readouts. Call readStocks(clauses) for
   actual living material, original experiment slots and source consumption rules.
6. Preserve clauses in multiple branches when they govern more than one biological boundary.
   A maternal-tissue rule can constrain both observation scope and a proposed offspring readout.
7. Return BankSource with the actual four child results, complete documents/coverage and accumulated
   unresolved source issues. Source role hints organize reading; original text governs interpretation.
   This complete BankSource is the input to reconstructBelief, predictExperiments and planPolicy.
   Those named stages own posterior, experimental answer and policy/cost conclusions after inventory.
