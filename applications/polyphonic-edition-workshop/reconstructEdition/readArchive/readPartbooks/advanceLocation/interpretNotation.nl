---
description: Judge one diplomatic notation statement without performing the score.
args:
  unit: NotationUnit
  selectedClauses: SourceClause[]
returns: NotationReading
---
1. Create a scoped NL NotationReading literal and directly await it on the actual unit and original
   selectedClauses. Its first numbered step locates this original event within its stated voice/section.
2. Its second step reads event order, written pitch rank, literal duration glyph, accidental and event
   kind. These are diplomatic symbols and source-described roles, not sounding pitches or elapsed time.
3. Its third step distinguishes an original note reading from a copied expansion or uncertain mark.
   Preserve a readable diplomatic form with its qualification in exact evidence; later arbitration owns its authority. Use unresolved only for missing interpretation,
   a genuinely unreadable decisive symbol returns meaning=null rather than a convenient number.
4. Its fourth step returns exact clause-keyed quotations supporting every decided field. No prior
   canonical timeline enters the scope; transposition and accidental carry belong to performance.
5. Return the actual awaited NotationReading. The parent binds original source identity and inserts
   these exact semantic fields; another stage owns authority and selection.
