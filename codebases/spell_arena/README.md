# Spell arena: finite application slice

`cast.nl` lets natlang interpret a spell into an ordered plan. The exact resolver checks the
world revision, actor IDs, integer limits, total energy, movement and occupied squares. It
commits the whole plan or returns the original world. The interpreter sees only portable
typed records. No TypeScript application host is needed.

The semantic leaf may return `plan`, `clarify` or `impossible`. Model quality is measured
separately from the resolver's safety. A later game loop can feed player and world events
through Fold; the finite cast already establishes the plan boundary for training cases.
