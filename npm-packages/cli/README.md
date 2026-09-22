# `@natlang/cli`

The native natlang command line. It installs content addressed `.nlpkg` archives and launches their declared executable targets.

```sh
npm install --global @natlang/cli
natlang --setup
natlang --help
natlang path/to/program.nl
natlang path/to/application
natlang summarize the available codebase functions
natlang --apps
natlang semantic-terminal
```

Multiple words that do not resolve to a path execute as an anonymous
`Lambda<{ files: Dict<ProjectFile> }, Text>` over the current directory's
top-level natlang functions and lazy file tree.

`natlang --setup` validates an explicit or PATH `llama-server` and, when needed,
asks before installing natlang's pinned, hash checked runtime in the user data
directory. It never replaces a system installation. Use `natlang --setup --yes`
in unattended environments. The first source or reducer-backed application run
offers the same setup when it has an interactive terminal.

Source paths require no package installation. A reducer-backed application
starts preparing its selected local server immediately, concurrently with
target initialization, and opens after readiness. Source execution through this
CLI also prepares the model immediately; only the lower-level embeddable model
session remains lazy until its caller invokes `prepare()` or `turn()`. The CLI
stops an owned server when the command exits.
`natlang --runtime status --json` reports discovery and compatibility details.
