# `@natlang/node`

Node.js host APIs for running natlang programs, building terminal applications,
managing a local model session, and creating native `.nlpkg` packages.
Install `@natlang/cli` for path based execution and optional package management.
The model API discovers and validates explicit, natlang managed, and PATH
llama.cpp runtimes. The CLI supplies interactive consent for managed downloads;
embedded hosts decide their own consent policy through `ensureRuntime`.
Sessions remain lazy by default; call `prepare()` to start and health-check the
managed model concurrently with other application initialization.
