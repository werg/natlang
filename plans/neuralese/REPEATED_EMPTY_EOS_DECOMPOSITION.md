# Repeated empty EOS handling

## Behavior

When a model reply ends immediately with EOS and contains no text, tool call,
generated token, or staged final value, the native agent gives the model one
ordinary recovery turn using the existing missing-result feedback. If the next
reply is the same immediate empty EOS and the call still has no result, the
agent ends the call as `model_incomplete: terminated_without_result`.

This is a model-incomplete outcome, not a transport failure or a timeout. The
agent must not infer EOS from an empty decoded string alone: it requires exact
provider evidence (`finish_reason: stop`, null/empty message content, no tool
calls, and zero completion tokens). Any text, tool call, generated token,
truncation, or other finish reason resets the consecutive-empty count.

## Decomposition

1. The native agent reads the decoded turn and the provider's exact raw response
   metadata. This is crisp protocol verification because the provider owns EOS
   and usage fields.
2. The agent maintains a consecutive count only while the call has no accepted
   result. The first empty EOS follows the existing feedback path unchanged.
3. After the second consecutive exact empty EOS, the agent emits a typed trace
   event and returns the explicit incomplete detail. The collector records this
   separately from infrastructure and resource-limit outcomes.
4. The scorer recognizes only this explicit detail as `model_incomplete`; it
   does not infer incompleteness from repeated empty-looking turns or elapsed
   time.

The model-facing prompt and result contract do not change. The existing recovery
message is sent once, and no expected answer or target is added to it.

## Verification

Focused tests cover two exact empty EOS replies, a single EOS followed by a
valid result, and empty replies without exact raw EOS proof. Frozen evaluations
must bind a new runtime and score version; an active packet or immutable runtime
is never edited in place.
