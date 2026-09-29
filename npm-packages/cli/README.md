# `@natlang/cli`

The natlang command line: build, check, and run natlang TypeScript applications,
call natural-language functions, and manage optional `.nlpkg` distribution
packages and model backends.

For current instruction adaptation APIs and implementation status, see the
[adaptation guide](../../docs/ADAPTATION.md).

```sh
npm install --global @natlang/cli
natlang setup                           # prepare the managed local model runtime
natlang doctor
natlang models anthropic                # list a provider's model IDs
natlang auth login anthropic            # sign in when the provider offers OAuth
natlang run path/to/application --provider anthropic --model MODEL_ID
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
`run`, `call`, `ask`, `setup`, and `doctor` share profile resolution and accept
`--profile`, `--provider`, and `--model`. A profile can select a managed local
model, an OpenAI-compatible endpoint, or a Pi provider. `natlang models` lists
providers and model IDs; `natlang auth login|status|logout` manages supported
OAuth credentials. See [model configuration](https://github.com/werg/natlang/blob/main/NATIVE_PACKAGES.md#authority-and-the-model-runtime)
for profile fields, precedence, provider options, and runtime controls.
# Adaptation status

The current adaptation APIs and remaining release gates are described in the
[adaptation guide](../../docs/ADAPTATION.md).
