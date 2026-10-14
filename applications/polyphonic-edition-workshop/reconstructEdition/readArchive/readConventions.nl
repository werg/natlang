---
description: Interpret the fictional score notation and edition-admission contract.
args:
  clauses: SourceClause[]
returns: NotationProtocol
---
1. Identify governing notation clauses and their explicit applicability. Read the duration glyph
   lexicon as supplied meanings; familiar notation outside this source supplies no extra rule.
2. Interpret each named voice's written-to-sounding transposition and the units of pitch ranks/ticks.
   Preserve exact witnesses for each duration and voice convention; conflicting assignments remain unknown.
3. Interpret accidental persistence and cancellation by written rank, voice and section occurrence.
   State the reset policy separately from pitch arithmetic so repetition does not inherit stale state.
4. Read the finite performance order and assign its original occurrence IDs, section IDs and order.
   A repeated section reuses its chosen written reading while applying the supplied reset convention.
5. Interpret editorial authority, ambiguity retention and evidential independence policy. Separate
   permission to make a correction from evidence establishing its actual target or operation.
6. Return durations, voices, occurrences, sectionTicks, accidentalPolicy, resetPolicy, editorialPolicy,
   original witnesses and unresolved premises. Numeric fields express interpreted source meanings;
   later exact arithmetic consumes them and cannot decide an unsupported convention.
