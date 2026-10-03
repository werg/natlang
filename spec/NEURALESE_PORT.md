# Neuralese port contract

Version 1, 2026-10-03. Normative for runtimes and model servers. Part of the
Neuralese section of [SPEC.md](SPEC.md). The model-side procedures are designed in
[plans/neuralese/S3_PORT.md](../plans/neuralese/S3_PORT.md); the source design is
[port mechanics and training](../plans/neuralese/sources/port-mechanics-and-training.md).

## Control tokens

Two special tokens delimit a soft block:

| Token | LFM2.5 ID slot |
| --- | --- |
| `<|neuralese|>` | `<|reserved_7|>` |
| `<|/neuralese|>` | `<|reserved_8|>` |

Each has its own trained input embedding. Other backbones register the same token
strings as special tokens. A model declares the dialect it reads and writes
([NEURALESE_DIALECTS.md](NEURALESE_DIALECTS.md)).

**Escaping.** Only the runtime (when rendering) and the write procedure (when
writing) produce these token IDs. Renderers tokenize content (messages, tool
outputs, file contents, string values) with special-token parsing disabled, so the
text `<|neuralese|>` in content becomes ordinary text tokens.

## Placement

A block may occur wherever the runtime shows or the model writes TypeScript or
values, inside the backbone's native chat template, with no new role:

- message content, including the opening declarations of a call;
- tool-call arguments, in particular the `code` string of `eval` and the `value` of
  `return_result`. For LFM2.5's Pythonic tool calls
  (`<|tool_call_start|>[eval(code="…")]<|tool_call_end|>`), the block sits inside
  the quoted string. Tool-call parsers recognise the control tokens there and do
  not treat the vectors as string characters;
- tool responses, where eval results and locals are listed.

The block ends at `<|/neuralese|>`. It does not end the message, the tool call, the
string, or the function. Everything that describes a block (its type, a function
signature, captures) is ordinary TypeScript before the opening token.

## Rendering (reference → model form)

When rendering a conversation, the runtime replaces each soft value it shows with
the token `<|neuralese|>`, `L` payload positions, and `<|/neuralese|>`, where `L`
is the stored block's length. It sends the payload as a content part referencing
the block ID (see Wire protocol). The server builds the input embedding sequence
by embedding the surrounding tokens and splicing the stored vectors (mapped into
the model's space by its adapter, if any) at the payload positions. Attention
masks, positions and lengths count every payload position.

## Writing (model form → reference)

When the model samples `<|neuralese|>` during decoding, the server switches to the
write procedure:

1. Run the shallow sketch recurrence (layers `1…k`) from the current context,
   one position at a time. At each position the stop head decides `continue` or
   `stop`; stop is masked before the first position unless empty blocks are
   allowed.
2. On `stop`, run layers `k+1…D` over the collected shallow states in one causal
   pass and project the completed states into the payload.
3. Store the payload as a new block (content ID, dialect, length, producer) and
   emit `<|/neuralese|>`.
4. Read back: restore the decoder cache to the position before the payload,
   prefill the payload and `<|/neuralese|>` through the full model, and resume
   ordinary decoding.

Restoring the cache covers attention keys and values and short-convolution state,
with positions reset to the committed sequence. Text sampling settings apply only
to text; the block's length is decided by the stop head alone.

**Hard maximum.** The runtime sets a maximum block length per server or request.
Reaching it closes the block, sets `truncated: true` on the stored entry, and is
recorded in the trace; it is not learned stopping.

**Parsing.** In the model turn returned to the runtime, each written block appears
as a content part with its ID, at its position in the text or tool argument. Before
compiling eval code, the runtime replaces each block with the reference expression
`__neuralese.value("nz1_…")` (or `__neuralese.body("nz1_…")` inside an `nl.with`
template), so the TypeScript checker sees ordinary code.

## Wire protocol

The model-turn request and response (`ts-host/src/contracts.ts`) gain a content
part, in both directions:

```json
{ "type": "neuralese", "id": "nz1_…" }
```

It may appear in message content arrays and in tool-call argument strings, which
are then sent as arrays of text and neuralese parts. A response reports, per
written block, its ID, length, `truncated`, and the stop decisions when tracing is
on.

A Neuralese-capable server exposes:

| Endpoint | Meaning |
| --- | --- |
| `GET /v1/neuralese/info` | Dialects spoken, width, dtype, maximum block length, whether `grad` sessions are available. |
| `PUT /v1/neuralese/blocks/{id}` | Store a block (safetensors body). The server verifies the ID against the content. |
| `GET /v1/neuralese/blocks/{id}` | Fetch a block. |
| `POST /v1/neuralese/blocks/{id}/pin` | Keep a block through garbage collection while a reference is live. |

A request containing neuralese parts sent to a server that does not declare a
matching dialect fails; the runtime reports `neuralese-unsupported-backend` or
`neuralese-dialect-mismatch`. There is no text fallback.

## Agreement

A server's cached execution must agree with recomputation on the same committed
sequence: identical greedy continuations and logits within the tolerance stated by
the parity suite (`conformance/neuralese/`).
