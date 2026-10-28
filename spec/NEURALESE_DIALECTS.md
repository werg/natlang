# Neuralese dialects

Version 1, 2026-10-03. Normative. Part of the Neuralese section of [SPEC.md](SPEC.md).

A dialect is a tag for the space a soft value's state lives in. It names a kind
of state, not one checkpoint: whether a reader may splice a state unchanged is
decided by the state's exact model identity, which every native state carries.

- **Tag.** A native state written by a model is in `native-state/<architecture>`,
  e.g. `native-state/lfm2` or `native-state/lfm3`. A state written through a configured learned
  semantic adapter (a compact `neuralese.semantic-state/1` message) is in
  `semantic-state/<adapter SHA-256>`. A server declares its tag first in the
  `dialects` of `/v1/neuralese/info`. Named tensor artifacts carry
  `native-tensor/<kind>` in store metadata; they are not soft values. The
  retired projected-vector tags (`nd:<name>@<version>`) name no current
  writer or reader.
- **Identity.** Every native state records its writer's checkpoint SHA-256,
  tokenizer fingerprint, embedding-weight SHA-256 and architecture. A reader
  splices a state's KV and hybrid state unchanged only when all four equal its
  own (`exact-native-state`), or under an explicitly requested
  `same-base-adapter-state` on the exact same base. Two checkpoints of one
  architecture share a dialect tag but not an identity.
- **Where it appears.** In the type (`Neuralese<T, D>`, with `D` defaulting to
  `DefaultDialect`), on every store entry, in the `.nz` header, and in a
  server's `/v1/neuralese/info`.
- **Compatibility check.** Loading, passing or sending a soft value whose tag does
  not match an explicitly named dialect is rejected (`neuralese-dialect-mismatch`).
  At a `DefaultDialect` slot, a native or semantic state of another tag is
  accepted only when the reader declares a consumption mode that can take it
  (`best-effort-text-retokenization`, or `learned-semantic-state-deltas` with
  the matching adapter); otherwise it is rejected with
  `neuralese-native-consumer-unavailable`. The value is then written again from
  its exact source with the reader's native writer, or converted with
  `convert(v, to)`. Values are never silently relabelled across dialects or
  identities.
- **Changing models.** A changed checkpoint writes states with a new identity.
  States written by the old checkpoint stay valid for readers with that exact
  identity; another reader needs explicit text re-encoding or a configured
  learned semantic adapter.
- **Adapters.** A model reads another model's state either by text
  re-encoding (the reader discards the foreign tensors and tokenizes the stored
  text itself; matching across tokenizers is best effort) or through a
  configured learned semantic adapter that decodes shared per-token codes into
  deltas for its own state (`learned-semantic-state-deltas`; a mismatch is
  `neuralese-semantic-adapter-mismatch`). The shared-space semantic adapter is
  a later training stage; none is trained yet.
- **Default.** `DefaultDialect` is bound by runtime configuration. In the
  TypeScript host it is the runtime's reader dialect: `neuralese.dialect` in the
  runtime options, else the write port's dialect (the two must agree), exposed
  as `NatlangRuntime.readerDialect()` (null for a text-only runtime). Hosts check
  it at startup against what their consumers read (a server's
  `/v1/neuralese/info`), not by a failing request. Arguments, results, `let`
  variables and `read` check the dialect each state was stored with
  (`ts-host/src/native/values.ts`).
- **Measurements.** Measurements tied to a model, such as the comparisons that
  enable law-based rewrites ([NEURALESE_REWRITES.md](NEURALESE_REWRITES.md)), are
  keyed by model and dialect version and re-run when either changes.
