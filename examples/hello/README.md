# hello

```sh
natlang run                          # plan the bundled messages.json (three decisions)
natlang run . -- '[{"from":"a@b","subject":"Hi","body":"Are you free at 3?"}]'   # one that drafts a reply
natlang call route.nl '{"from":"a@b","subject":"Invoice","body":"Can you resend it?"}'
natlang dev                          # rerun after every edit; Ctrl+C stops
natlang test                         # check the saved examples
natlang dashboard --open             # results, traces and model settings in the browser
```

**types.ts** is the data: a `Message` whose body is `Untrusted<string>` (the model reads it as data and never follows
it), the `Route` a message can take, and a `Reply` whose property the runtime checks on every draft.

**route.nl** is a natural-language function: a typed signature and the instructions a model carries out. Its
`readout: decision` makes the model reason, then score the three routes, and `decide` returns their probabilities.

**main.ts** is the program: it routes every message at once, drafts a reply with an inline `nl<Reply>` function only
where one is needed, and keeps each decision's confidence. `main` is what `natlang run` calls.

**examples.json** saves inputs with their expected results; `natlang test` runs them on the real model.
**messages.json** is the sample input. Callable folders hold what a function may call, created when needed: **route/**
for `route.nl`, **natlang.d/** for the inline `nl` in main.ts (`.ts` for exact steps, `.nl` for judgments).

`natlang check` checks the program without calling a model. Every call prints its call ID on stderr: `natlang traces
show CALL_ID` shows the steps the model took. The first run prepares the selected model: the local one is a one-time
download (1.8 GB), or choose a hosted model on the dashboard's Models page. AGENTS.md is the guide for coding agents.
