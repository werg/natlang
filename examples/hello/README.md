# hello

```sh
natlang run                          # answer.nl on the sample input
natlang run . -- "My own text"       # or: natlang call answer.nl "My own text"
natlang dev                          # rerun after every edit; Ctrl+C stops
natlang test                         # check the saved examples
natlang dashboard --open             # results, traces and model settings in the browser
```

**answer.nl** is a natural-language function: a typed signature (`args`, `returns`) and instructions that a model
carries out. Edit the instructions to change its behavior. **examples.json** saves inputs with their expected results, and
`natlang test` runs them on the real model. The sample input is the `message` argument's default in **natlang.json**.
Helpers that answer.nl may call go in the **answer/** folder; **package.json** makes TypeScript files ES modules.

`natlang check` checks the source without calling a model. Every call prints its call ID on stderr: `natlang traces show
CALL_ID` shows the steps the model took. The first run prepares the selected model: the local one is a one-time download
(1.8 GB), or choose a hosted model on the dashboard's Models page. AGENTS.md is the guide for coding agents.
