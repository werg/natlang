# hello

```sh
natlang run                          # answer.nl on the sample input
natlang run . -- "My own input"      # or: natlang call answer.nl "My own input"
natlang dev                          # rerun after every edit; Ctrl+C stops
natlang test                         # check the saved examples
natlang dashboard --open             # results, traces and model settings in the browser
```

**answer.nl** is a natural-language function: a typed signature (`args`, `returns`) and instructions that a model
carries out. Edit the instructions to change its behavior. **examples.json** saves inputs with their expected results, and
`natlang test` runs them on the real model. The sample input is the `input` argument's default in **natlang.json**.

`natlang check` checks the source without calling a model. The first run prepares the selected model: the local one is a
one-time download (about 1.9 GB), or choose a hosted model on the dashboard's Models page.
