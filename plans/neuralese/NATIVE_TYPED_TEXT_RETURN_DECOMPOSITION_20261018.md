# Native typed text returns

## Contract

A final assistant text reply is the returned value only when it can be coerced to
the invocation's declared return type. For `Neuralese<string>`, nonempty plain
text is an exact string value that must pass through the same typed text writer
as `return_result(value)` and eval returns. It must finish the invocation with a
real stored block and an observed typed host result. A reply that is not a
return value remains a nonterminal attempt; it must not be classified as a
successful tool action merely because it emitted no tool calls.

The proof for a returned text block binds the final raw assistant response,
its model-turn node, the block-write event and body digest, the declared result
type, and the completed invocation host result. A prior plain-text attempt with
no completed typed result is not a writer. Existing `return_result` and eval
materialization keep their current source identity and proof rules.

## Decomposition

1. **Natural-language policy:** the call's declared return type determines
   whether final text is a result. `done` continues to return an already staged
   value; prose, tool-call markup and incomplete replies do not become typed
   values. This judgment belongs to the callable's type and the model's actual
   final reply, not a new heuristic or task-specific rule.
2. **Crisp runtime boundary:** exact type coercion and `Neuralese<string>` block
   creation use the existing asynchronous `coerceReturn` and text-writer path.
   The runtime must await this before checking missing-result feedback or
   finishing the invocation. The writer records the source as a terminal
   assistant text return, distinct from `return_result` and eval.
3. **Crisp materialization verifier:** the teacher materializer accepts this
   source only when the same invocation has a completed host capture of the
   exact block, the write is linked to the final model-turn node, the raw final
   text maps to the exact body by the runtime's documented trim operation, and
   there are no later turns or ambiguous writes. It records a separate typed
   write receipt on the decision; it never invents a tool call.
4. **Crisp decision classification:** a call-less response followed by a
   same-invocation continuation because no result was staged is a failed
   completion attempt. Other call-less turns retain their existing
   classification. Completed typed text returns are successful and are not
   labeled failed.
5. **Training lowering:** the exact assistant text remains the model target,
   including its original whitespace. The compiler adds a separately hashed
   assistant-text-result sidecar with the exact trimmed body span and source
   receipt. The shared write-site extractor authenticates it as a real writer;
   trajectory training uses the original conversation prefix and the receipt's
   body as the source for downstream state readers. It does not invent a tool
   call or replace the assistant target with a synthetic marker.
6. **Measurement:** no new prompt or tool schema is introduced. The changed
   finish behavior is model-facing and is recorded as measurement debt until
   sampled in the next own-model teacher window.

## Verification

Focused tests cover a final `Neuralese<string>` text reply materializing a
block, exact host capture and source receipt; rejection of unbound or
nonterminal text as a writer; unchanged plain-string and JSON returns; `done`
preserving a staged result; and recovery classification for a no-call response
that receives missing-result feedback. Materializer tampering tests change the
terminal turn, body, block reference, host capture and invocation lineage.
Real runtime/teacher samples measure the updated completion behavior before
production adoption; mechanism tests alone are not learned-quality evidence.
