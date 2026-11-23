# Working on this natlang program

This program automates a piece of judgment work with a small local model. TypeScript holds the types, the data and
every exact step; each judgment is a typed natural-language function the model runs. Read `main.ts` first: it is the
whole program in one screen.

## Files

- `types.ts`: the data. `Untrusted<string>` marks outside text (the model reads it as data and never follows it);
  `Is<string, "...">` is a property the runtime checks on every value and repairs when it fails.
- `route.nl`: a named natural-language function. YAML front matter declares its typed signature; the body says what
  to decide. `readout: decision` makes the model reason, then score each allowed value: calibrated
  (`decide(route, message)` gives you the probabilities).
- `main.ts`: TypeScript that maps `route` over the messages, and an inline function, ``nl<Reply>`Draft a reply to
  message for its reader.` ``, called only where a reply is needed. An inline `nl` takes its call's arguments under
  the names the call writes (`draft(message)` gives it `message`) and captures the other variables its instructions
  name. `main` is what `natlang run` calls.
- `natlang.json`: the package. `targets.main` runs main.ts's `main`; its `messages` argument has a type from
  types.ts, checked before the run. `include` lists what the package holds, folders included before they exist.
- `examples.json`: cases with expected results; `natlang test` runs them on the model.
- Callable folders (create them when needed): `route/` holds what `route.nl` may call, `natlang.d/` what the
  inline `nl` in main.ts may call; `.ts` for exact steps, `.nl` for judgments.
- Generated (do not edit): `*.d.nl.ts` and `.natlang/`.

## The loop

1. `natlang check --json`: types, signatures and call sites, without a model. Fix every diagnostic first; each one
   says what to write instead.
2. `natlang call route.nl '{"from":"a@b","subject":"Invoice","body":"Can you resend it?"}' --json`: one real call.
   `natlang run` runs the whole program on messages.json, whose messages need no reply; `natlang run . -- '[...]'`
   plans your own messages as JSON, and drafts a reply for each one that needs it.
3. `natlang test --json`: every case in examples.json. When you change behaviour, add a case first.
4. `natlang traces show CALL_ID`: every step the model took (each call prints its ID on stderr).
   `natlang traces fix CALL_ID --expected JSON` turns a wrong result into a saved case.

Exit codes: 0 ok; 1 the program is wrong; 2 usage error; 3 the model is unavailable.

## Growing the program

- A new judgment is a new typed function: a `.nl` file when it is reused or has its own cases, an inline `nl<T>`
  when it belongs to one place in the code. Give it the narrowest result type that fits: a literal union with
  `readout: decision` when the answer is one of a few values, a record when it has parts.
- An exact step (counting, sorting, parsing, joining records, arithmetic) is a TypeScript function; a `.nl` function
  that needs it calls it from its callable folder.
- A property of a result goes in its type (`Is<T, "...">`), and outside text in `Untrusted<T>`. The runtime then
  checks and repairs them, with no instructions needed.
- Refining until good enough is `iterateOn`: ``iterateOn(revise, draft).until(nl`draft answers every question in
  message`)``. The step function returns the next state, and a natural-language stopping check is allowed.
- The same function under another policy is `fn.in(context)`; on a stronger model it is `fn.on({ model: NAME })`.
  Gate on `decide`'s confidence and send only the unsure cases there.
- A decision with both a code rule and a judgment uses `pluggable({ crisp, nl }, setting)`. In `shadow` mode both
  run, and their disagreements are recorded.
- Repeated inputs compile to code: `natlang specialize` turns a function's recorded calls into guarded crisp cases
  that answer without the model, and hands every other input back to it.
- The program improves from its runs: `natlang tune` trains the local model's weights on this program's examples (it
  needs a native student model; `natlang doctor` says whether one is set up), and `natlang improve init --export plan`
  writes an improvement case: give it train, validation and test cases, and `natlang improve CASE.json --out DIR
  --model MODEL` searches for better source against them.

## The model

On exit code 3, ask the user before running `natlang setup --yes` (a one-time 1.8 GB download), or use a hosted
model: `--provider ID` (`natlang models`) or `NATLANG_SERVER=URL NATLANG_MODEL=ID`. `natlang doctor --json` reports
the setup.

The natlang-authoring skill (`natlang skills install`) has the full language: composition, transactions over
folders, Neuralese values, recursion and decisions.
