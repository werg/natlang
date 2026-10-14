---
description: Inventory original definition units and preserve each actual interpreted contribution.
args:
  documents: ProtocolRecord[]
returns: EvidenceIndex
---
1. Keep documents immutable with original id/path/text/sha256. Inventory the finite original clause
   universe in those documents. Source IDs and bracketed clause markers are provenance, not meanings;
   exact data JSON remains trial evidence rather than a new definition merely because it contains numbers.
2. Declare a local ReadingInventory with units:DefinitionReadingUnit[] and unresolved:string[]. Directly
   invoke a scoped NL ReadingInventory literal on documents. Its numbered policy identifies each
   definition-bearing original clause, preserves its exact original clause ID and quotation, and names
   one local focus describing the semantic distinction to interpret. Include question, inclusion,
   anchor, interval, numerator, alias, artifact binding, edit scope and intervention-permission clauses.
3. The inventory preserves historical/archive and implementation ownership distinctions rather than
   deciding which instruction should win. Where one clause contains several independent definitions,
   give each an original-span unit and distinct focus; keep common governing clause IDs as support.
   Record nondefinition evidence and unresolved source identity in inventory.unresolved only when they
   leave a necessary inventory role unknown. Do not invent a definition for every numeric data row.
4. Consume the actual inventory return. Require every sourceId/clauseId/exactQuote to identify an
   original document clause and nonempty exact substring. Require unique unitIds and exact original
   supporting clause membership. A malformed inventory boundary remains an explicit unresolved
   inventory finding, not a guessed semantic row. Retain all valid original units for actual interpretation.
5. Initialize DefinitionFrontier with those units, original documents, completed:[], definitions:[] and
   actual inventory findings. Invoke iterateOn with advanceDefinitionUnit and the structural predicate
   pendingUnits.length===0. Every step consumes one original pending unit; completed entries are carried
   forward. No elapsed-time or arbitrary retry count substitutes for this finite measure.
6. Consume the actual settled frontier. An accepted unit contributes its actual reading; a semantically
   unresolved unit contributes its reason, and a transport/type/provenance failure contributes its actual
   failure. Never erase earlier accepted readings because a later unit failed. Do not call completed units
   again when continuing a partially processed frontier.
7. Return documents unchanged, actual definitions, completed as unitResults, and actual unresolved
   obligations. This inventory derives neither trial plans nor repairs. Definition meaning belongs to
   interpretDefinition; target meaning belongs to interpretQuestion after the index exists.
8. An empty or incomplete accepted index is a visible prerequisite failure for later stages. Preserve
   which original units failed so a reader repair can target those units without recreating a raw-document
   semantic table in a downstream stage.
