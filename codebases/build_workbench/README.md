# Build workbench: serial execution slice

`build.nl` is a natlang dependency loop. It computes the goal's dependency
closure, asks the model to choose among ready tasks, checks that choice again,
and stops when the goal is built, a command fails, or no task can become ready.
The work limit is derived from the task count; there is no fixed episode or
model-turn limit in this application.

`BuildWorkspace` in `applications/build_workbench.mjs` is the native Node host
object used by the crisp `advance` helper. Tasks carry an authored argv vector,
declared input paths and declared output paths. The adapter executes argv
without a shell, verifies inputs and outputs stay within the workspace,
refuses preexisting outputs, rejects changed inputs, and records input/output
digests and process outcomes. A timeout or signal is an unknown outcome because
the process may already have changed files. The native trace records host
observations, including failures that occur after a host mutation.

This is a **trusted command** interface, not a process sandbox. The adapter
cannot prove a command did not read another file, use the network, or write an
undeclared path. For that reason there is no cache yet: a hash of the declared
inputs alone would be unsound. A sound cache needs an execution profile that
enforces or observes the complete read set and toolchain/environment identity,
and checks stored output content on retrieval. A run that leaves output after
failure needs reconciliation in a fresh workspace before retry. External
effects of an aborted native call cannot be rolled back by the natlang runtime.

The integration test runs actual Node subprocesses through the natlang codebase
and checks a successful graph, command failure, cycle, changed input, stale
output and trace evidence. It uses a scripted model callback to verify the
interface. Teacher behavior, semantic task selection, cache soundness and
repair proposals remain separate gates.
