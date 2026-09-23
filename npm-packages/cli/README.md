# `@natlang/cli`

The natlang command line: build, check, and run natlang TypeScript applications,
call natural-language functions, and manage optional `.nlpkg` distribution
packages and the local model runtime.

```sh
npm install --global @natlang/cli
natlang setup                           # prepare the managed local model runtime
natlang doctor
natlang run path/to/application         # a directory with natlang.json, or a TS entry module
natlang check path/to/project
natlang call path/to/function.nl --inputs inputs.json
natlang ask summarize the functions in natlang.d
natlang apps
natlang package install application.nlpkg && natlang run application-name
```

`natlang run` compiles the project with `natlang build` and calls the entry's
exported `main(context)` with the configured model and runtime. `natlang setup`
validates an explicit or PATH `llama-server` and, when needed, asks before
installing natlang's pinned, hash-checked runtime in the user data directory;
`natlang setup --yes` approves it unattended. The CLI starts an owned model
server only for commands that need one and stops it when the command exits.
`natlang runtime status --json` reports discovery and compatibility details.
