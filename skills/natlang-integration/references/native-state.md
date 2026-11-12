# Native Neuralese state

A native Neuralese server (a natlang native model, `natlang setup --native`) can carry values between calls as model
state instead of text. In source this is the type `Neuralese<T>`; the runtime handles the rest. This page is what a
host needs to run and pass such values. Native learning (graph gradients, adapters, optimizer steps), RL sampling
receipts and training capture are research interfaces documented in the repository's `docs/NATIVE_RUNTIME_NOTES.md`.

## Wiring a native server

- Bind the server's writer and reader with `nativeStateRuntimeOptions({ endpoint, store, headers? })`. A store alone
  archives state but cannot produce it: `neuralese-native-text-write-unavailable` means the server lacks the text
  writer; use a server that has it, never a stand-in embedder. A host that only consumes state may pass
  `{ write: 'available' }` and must not claim it can create new state.
- The store must persist native artifacts (`neuralese-native-archive-unavailable` otherwise).
- `natlangRuntime({ programRoot })` does this for a native profile and starts its server; `natlang serve` keeps the
  server loaded between commands.

## Passing values

- A `Neuralese<T>` value is an opaque typed reference. Pass the value itself to the next call, a capture or a typed
  result; never rebuild a reference from its artifact ID, and never return an ID or marker as answer text. For a
  string slot that reports `type-mismatch`, read the value through a function that returns the requested type.
- In conversation history, keep `{ type: 'native_state', id }` parts structured, including inside tool-call arguments.
  Only a complete canonical `nzs1_` ID refers to a state; a shortened one is ordinary text. On
  `neuralese-unknown-native-state`, restore that artifact into the store and retry the same request unchanged.
- A value from another checkpoint is read through its text (`best-effort-text-retokenization`); only identical model
  identities splice raw state. `neuralese-native-consumer-unavailable` means the reader does not declare text
  re-encoding: use one that does, or rewrite the value with the target model's writer.
- `neuralese-native-empty-unavailable`: load the native library with its archived empty state.
- A function declared `returns: Neuralese<string>` turns its final text reply into state through the configured
  writer; a reply that leaves the call without a result is incomplete, not success.

## Student prompt rendering

Set `promptRendering: "program-header"` in the model profile (or the runtime's `model` settings) for students trained
beyond the full-prompt warm-up, matching the checkpoint's `prompt_rendering` declaration; other models default to
`"full"`. Program-header requests keep the signature, types, instructions, inputs, skills, call-specific declarations
and trajectory, and omit the generic interpreter prose and native tool schemas (the native tools still execute).
An application that calls the student directly with its own tools still sends their schemas, and serving keeps them.

## Learned compaction

The native agent's `compaction` setting (agent option, `neuralese.compaction`, or `NATLANG_NEURALESE_COMPACTION`)
selects `crisp` (text cut-offs, the default) or `nl`: on a server advertising `operator/1`, cut-off tool outputs,
argument listings and compacted history also carry a Neuralese summary state. The full value always stays reachable
from eval, `read_page` and `transcript`.
