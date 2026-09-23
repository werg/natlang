# Experiment laboratory

The laboratory has two natlang entry points. `design.nl` chooses cases from a
supplied, bounded menu; `report.nl` interprets exact counts, repeatability, and
outstanding semantic review. The host records one trial for every planned
candidate, case, and attempt, including exceptions and missing results, with
trial seeds derived from logical case and attempt IDs and separate model and
world seed namespaces.

The interactive laboratory is Studio's P16 application (`ts-host/studio/`). The
Python host that ran these programs against merge, type, dependency-plan, and
migration backends was removed with the Python runtime; these programs remain as
teacher corpus. Equal digests across repeats measure observed repeatability,
not correctness or convergence, and a trial's semantic quality stays pending
until review.
