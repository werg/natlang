# Neuralese dialects

Version 1, 2026-10-03. Normative. Part of the Neuralese section of [SPEC.md](SPEC.md).

A dialect is a lightweight version tag for the space soft values live in. Dialect
stability is not a goal while Neuralese is young: when training changes a model's
space, the version is bumped.

- **Tag.** `nd:<name>@<version>`, e.g. `nd:natlang@1`. It names the width, dtype,
  normalisation and control-token strings of the space. `nd:natlang@1` is the tag
  of the first S3-trained model.
- **Where it appears.** In the type (`Neuralese<T, D>`, with `D` defaulting to
  `DefaultDialect`), on every store entry and `.nz` block, and in a server's
  `/v1/neuralese/info`.
- **Compatibility check.** Loading, passing or sending a soft value whose tag does
  not match the expected dialect is rejected (`neuralese-dialect-mismatch`). The
  value is then regenerated from its exact source or converted with
  `convert(v, to)`. Values are never silently reused across dialects.
- **Bumping.** A model release whose space differs from its predecessor's gets a
  new version. Values written for the old version stay valid for models that
  speak it.
- **Adapters.** A model of a different width that must speak an existing dialect
  (for example a small browser model and a larger server model sharing one) ships
  input and output maps between its hidden width and the dialect's space. No other
  model needs adapters.
- **Default.** `DefaultDialect` is bound by program configuration
  (`NeuraleseConfig` in [neuralese.d.ts](neuralese.d.ts)). In the TypeScript
  host it is the runtime's reader dialect: `neuralese.dialect` in the runtime
  options, else the write port's dialect (the two must agree), exposed as
  `NatlangRuntime.readerDialect()` (null for a text-only runtime). Hosts check it
  at startup against what their consumers read (a server's
  `/v1/neuralese/info`), not by a failing request. Arguments, results, `let`
  variables and `read` check the dialect each block was stored with
  (`ts-host/src/native/values.ts`).
- **Measurements.** Measurements tied to a model, such as the comparisons that
  enable law-based rewrites ([NEURALESE_REWRITES.md](NEURALESE_REWRITES.md)), are
  keyed by model and dialect version and re-run when either changes.
