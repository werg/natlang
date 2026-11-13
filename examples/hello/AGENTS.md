# Working on this natlang program

natlang is TypeScript with natural-language functions. `answer.nl` is one: YAML front matter declares its typed
signature (`args`, `returns`), and the body is instructions that a model executes. The signature is checked like
TypeScript; the instructions' behavior is checked by running examples on the model.

## Files

- `answer.nl`: the function. Edit the instructions to change behavior, `args`/`returns` to change the contract.
- `examples.json`: saved cases, `[{ name, function: "answer.nl", inputs: { message: ... }, expected }]`, compared exactly.
- `natlang.json`: the package. `targets.main` runs `answer.nl`; its `message` argument's default is the sample.
  `include` lists what the package holds (`answer/` included).
- Generated (do not edit): `*.d.nl.ts` and `.natlang/`.

## The loop

1. `natlang check --json`: types, signatures and call sites, without a model. Prints
   `{ ok, diagnostics: [{ file, line, column, code, message, hint? }], functions: [{ source, signature }] }`; fix every
   diagnostic before calling.
2. `natlang call answer.nl "some text" --json`: one real call; arguments fill parameters in order. stdout is the
   result as JSON; stderr names the call: `natlang: call CALL_ID (natlang traces show CALL_ID)`.
3. `natlang test --json`: every example in examples.json, with expected and received values and each one's `call_id`.
4. `natlang traces show CALL_ID`: each step the model took, to see why a result is wrong. `natlang traces fix CALL_ID
   --expected JSON` saves that call's inputs with the corrected result into examples.json.

Exit codes: 0 ok; 1 the program is wrong (diagnostics, failed examples, a call that failed); 2 usage error (bad flags,
arguments or paths); 3 the model is unavailable (runtime not installed, endpoint unreachable, missing credentials).
Model calls take seconds on a hosted model and can take minutes on a local CPU, so check first and call deliberately.
When you change behavior, add or update an example and run `natlang test`.

## The model

On exit code 3, ask the user before running `natlang setup --yes` (a one-time download of 1.8 GB), or use a hosted
model: `--provider ID` (`natlang models` lists them) or `NATLANG_SERVER=URL NATLANG_MODEL=ID` for an
OpenAI-compatible endpoint. `natlang doctor --json` reports the setup.

## Writing natlang

- Exact work (arithmetic, parsing, sorting, I/O) belongs in TypeScript; judgment belongs in instructions.
- Put constraints in types: a string-literal union or a record in `returns` beats a sentence asking for a format.
- `readout: decision` (in answer.nl) makes the model score each value of a finite `returns` (a literal union or
  `boolean`) instead of reasoning step by step: fast and calibrated, for a choice that gates what happens next. Such a
  call gets no tools, so a function that must call helpers or compute first leaves it out.
- Helpers the function may call go in its companion folder, `answer/` (`answer/lookup.ts`, `answer/summarize.nl`).
- Number the steps of a multi-step instruction and name the data each step uses; the executing model may be small.
- Language reference and patterns: the natlang-authoring skill (`natlang skills install` adds it to Claude Code or Codex).
