# `@natlang/node`

The Node.js runtime and compiler for natlang: `nl`, `iterateOn`,
`createNatlangRuntime`, named `.nl` loading, `buildProject` / `checkProject`,
folders, services, traces, `EventLoop` and terminal utilities, the managed local
model session, and `.nlpkg` package APIs. Install `@natlang/cli` for the
`natlang` command.

The model API discovers and validates explicit, natlang-managed, and PATH
llama.cpp runtimes. Sessions stay lazy until `prepare()` or the first model turn.
