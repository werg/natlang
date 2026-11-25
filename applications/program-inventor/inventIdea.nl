---
description: Invent one completely new natlang program idea that differs from every idea in the atlas.
model: inventor
args:
  atlas: string
  rejected: RejectedIdea[]
returns: IdeaCard
---
atlas lists every program idea natlang already has: the task families it trains on, its applications, design briefs,
examples, dataset sources and the ideas this process already accepted, one line each. rejected holds the ideas this
round already turned down, each with the reason.

Invent one program idea that is completely new: it differs from every atlas entry and every rejected idea in at least
two of these four respects, and a change of subject matter alone (another industry, other names or numbers) counts as
none of them:
- who runs it and for what repeated job;
- the judgments of meaning at its core;
- the shape of its computation: how exact steps and judgments feed each other, what state is carried, where results
  combine;
- the kind of input data it reads.

1. Read the whole atlas. Note which users, judgments, computational shapes and kinds of input it covers, and where it
   is crowded. Read the ideas this process already accepted as one group: when they share a pattern (the kind of
   user, the role their judgments play, the kind of exact computation that follows), your idea lies outside that
   pattern in at least two of those three.
2. Think of repeat work a real person or team does today by hand, with brittle scripts or with a frontier agent loop,
   whose correct result needs several different judgments of meaning combined by exact steps. Prefer work whose
   difficulty is in meaning: what a clause commits someone to, whether two descriptions are the same thing, which
   later statement overrides an earlier one, what a change breaks.
3. Design it as a natlang program: TypeScript owns the types, data and every exact step (parsing, ordering, counting,
   arithmetic, state transitions, graph walks); each judgment of meaning is one narrow typed function. judgments
   lists at least three judgments, each asking a different question, each with the wrong reading a careless or
   keyword-driven reader would give. The program's answer must change when any one judgment is answered wrong.
4. Give the computation real depth: at least five functions, a call chain at least three deep, a judgment that reads
   a passage of several sentences and weighs it, a judgment that reads the results of two earlier judgments or state
   carried across items, at least two rules or stages whose order or precedence decides the outcome, and an exact
   algorithm at the core (a graph walk, a fixed point, a search, an enumeration). Each judgment maps words to meanings
   and referents; the code computes positions, ordinals and counts from them. constructs names at least three of the
   listed constructs the design genuinely needs.
5. One input instance is one real working session of that user: 1,000 to 2,000 words of text people wrote, in 15 to 30
   items a judgment reads, with the evidence for each reading spread through the text the way it is in real work. It
   has exactly one correct output that an exact checker can decide (a record, a list of records, a set, a final
   state). inputs states the size of one instance in items and words; the build is held to it. whyKeywordsFail
   explains why no rule over surface cues reaches it.
6. nearest names the atlas entries closest to your idea, each by the name its line starts with, and difference says for
   each how yours differs in the four respects above. If the honest difference is only subject matter, go back to
   step 2 and choose other work.
