# Natlang scheduler

`plan.nl` reads an exact calendar snapshot and complete feasible schedules,
then asks natlang to choose among them using the user's soft preferences. The
host validates the complete candidate again under an expected revision before
committing. A new block event increments the calendar revision, invalidates
stale proposals, and reports conflicts with existing scheduled tasks.

Times require explicit UTC offsets and are converted to integer UTC minutes.
The host handles duration, dependencies, windows, fixed commitments and
non-overlap. Candidate enumeration is capped and reports truncation. A real
calendar adapter needs conditional writes or an explicit reconciliation path;
the local fixture does not claim external calendar effects.
