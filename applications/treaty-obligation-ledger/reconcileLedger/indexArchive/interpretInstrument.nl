---
description: Separate an instrument's claimed act, issuer, adoption and temporal scope.
args:
  document: SourceDocument
  clauses: Clause[]
returns: Instrument[]
---
1. Read clauses belonging to document. Identify distinct issued instruments rather than assuming one
   file equals one legal act; reciprocal letters in one file remain separate claims.
2. For each instrument identify issuer and kind from its declared function. A draft, adopted amendment,
   accession reservation, operational order and evidentiary certification have different consequences.
3. Extract adoption and publication times only where supported. Preserve absent dates as null; null
   means unspecified for adoption/publication metadata, not day0. For an explicitly open effective
   interval, use a null end; ambiguous effectiveness goes in unresolved. Convert inclusive integer-day
   ends to the equivalent exclusive boundary.
4. Record the instrument's claimed effective interval separately from its authority. The authority
   branch later intersects that interval with the issuer's competence and can reject the claimed scope.
5. Identify archive availability independently. A certification can prove a past event only in a
   snapshot that has received it; retrospective evidence is different from retroactive legislation.
6. Return one witnessed Instrument per issued object with its clause IDs. Do not turn an assertion
   of power into proof of power, and do not silently promote a draft to its later adopted version.
