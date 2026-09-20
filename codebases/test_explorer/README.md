# Dependency-plan test explorer

Natlang selects scenarios and interprets checked observations. The host freezes
case inputs, runs each trial in a fresh runtime and verifies a graph invariant:
every emitted task exists, occurs once, follows its dependencies, and all tasks
which can be ordered appear before the planner declares completion. A semantic
failure explanation never becomes the oracle. Shrinking removes tasks and reruns
the same invariant; it only reports a smaller case when the same violation code
reproduces. Model identity, seed, source revision and trace digest accompany trials.
