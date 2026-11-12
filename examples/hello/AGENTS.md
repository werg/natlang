# Working on this natlang program

natlang is TypeScript with natural-language functions. `answer.nl` is one: YAML front matter declares its typed
signature (`args`, `returns`), and the body is instructions that a model executes. The signature is checked like
TypeScript; the instructions' behavior is checked by running examples on the model.

## Files

- `answer.nl`: the function. Edit the instructions to change behavior, `args`/`returns` to change the contract.
- `examples.json`: saved cases, `[{ name, function: "answer.nl", inputs: { input: ... }, expected }]`, compared exactly.
- `natlang.json`: the package. `targets.main` runs `answer.nl`; its `input` argument's default is the sample.
- Generated (do not edit): `*.d.nl.ts` and `.natlang/`.

## The loop

1. `natlang check --json`: types, signatures and call sites, without a model. Fix every diagnostic before calling it.
2. `natlang call answer.nl "some input" --json`: one real call; arguments fill parameters in order.
3. `natlang test --json`: every example in examples.json; exits 1 if any fails, with expected and received values.
4. `natlang traces list` and `natlang traces show CALL_ID`: each step the model took, to see why a result is wrong.

Model calls take seconds on a hosted model and can take minutes on a local CPU, so check first and call deliberately.
When you change behavior, add or update an example and run `natlang test`.

## The model

If a command reports that the local runtime is not installed, ask the user before running `natlang setup --yes` (a
one-time download of about 1.9 GB), or use a hosted model: `--provider ID` (`natlang models` lists them) or
`NATLANG_SERVER=URL NATLANG_MODEL=ID` for an OpenAI-compatible endpoint. `natlang doctor --json` reports the setup.

## Writing natlang

- Exact work (arithmetic, parsing, sorting, I/O) belongs in TypeScript; judgment belongs in instructions.
- Put constraints in types: a string-literal union or a record in `returns` beats a sentence asking for a format.
- Helpers the function may call go in its companion folder, `answer/` (`answer/lookup.ts`, `answer/summarize.nl`).
- Number the steps of a multi-step instruction and name the data each step uses; the executing model may be small.
- Language reference and patterns: the natlang-authoring skill (`natlang skills install` adds it to Claude Code or Codex).
