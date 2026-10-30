# hello

Run these commands from this directory:

```sh
natlang run
natlang run . -- "My own input"
natlang dev
natlang check
natlang test
natlang dashboard --open
```

Edit **answer.nl** to change the instructions, **sample.json** to change the default input, and **examples.json** to save input/expected-output tests. The dashboard can also save tests from the function form. Model tests execute real calls on your selected model.

The arguments and return type in answer.nl are the function's contract. Imports and call sites are checked before execution; the model's behavior still needs examples and tests. Instructions can use the declared input and callable helpers in the companion answer/ folder. Add typed services when the model needs outside capabilities.

The first run prepares your selected model. Local setup may download a model; you can instead configure an API provider in the dashboard's Models page. Runs reuse saved prompt openings automatically. Use a model profile to choose another backend.

When a run fails, the dashboard shows its error and links calls to source and traces.
