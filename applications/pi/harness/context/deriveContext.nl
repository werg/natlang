---
description: Context derivation (pi-durable context.ts deriveRange, contribute, selectActive, orderToolResults, missingResult, leadWithSystem; prompt.ts replaySections; pi-ai getCurrentTools; spec §2.1). The natural-language implementation of the pluggable context; the crisp one is pi-durable's.
args:
  head: EntryRecord | null
  entries: EntryRecord[]
returns: ContextView
---
Derive a conversation's model context from its raw active range. head is the newest head marker at or before the tail
(null when there is none). entries is every committed entry from head.head (or from the first entry) through the
tail, oldest first, older head markers included. The values in texts are exact: copy them, never retype them.
Work in eval: the steps are exact and the lists can be long.

1. Edits. edits = an empty map from entry ID to edit. For every entry in entries, oldest first, and every edit in its
   edits (if any), in order: edits[edit.target] = edit. A later edit of the same target replaces the earlier one.
   Older head markers count here too.
2. Active entries. With no head: active = entries. With a head: active = [head, then every entry of entries that has
   no head field], in entries order. (The head marker appears once, first; older head markers drop out.)
3. Contributions, one list per entry of active, in order. Take edit = edits[entry.id]:
   - edit.action "omit": the contribution is [];
   - edit.action "replace": start from edit.messages;
   - no edit: start from entry.model, or [] when the entry has no model.
   Then remove every assistant message whose stopReason is one of texts.EXCLUDED_STOP_REASONS.
4. Order tool results. all = every contribution concatenated in order. ordered = []. Walk all by position i:
   - a message with role "toolResult" is skipped where it stands;
   - any other message is pushed to ordered;
   - after pushing an assistant message whose content has toolCall items: look at the messages after position i up
     to (not including) the next assistant message; for each toolResult among them, remember the first one per
     toolCallId. Then, for each toolCall of that assistant message in content order, push its remembered result, or,
     when there is none, the synthesized missing result { role: "toolResult", toolCallId: call.id, toolName: call.name,
     content: [{ type: "text", text: texts.MISSING_RESULT_TEXT }], isError: true, details: { reason: "missing_result" },
     timestamp: <the assistant message's timestamp> }.
   Results that match no call, and second results for the same call, are dropped this way.
5. Lead with the system message. i = the index of the first message in ordered whose role is not "user". If i > 0 and
   ordered[i] is a system message, move it to the front (the messages before it keep their order). Otherwise ordered
   stays as it is. messages = the result.
6. Shown sections. Replay the system messages of messages in order into an ordered map key -> text: for each
   [key, value] of message.sections (if any), in order, null deletes the key, a string sets it; setting a key that is
   present keeps its position, a new or deleted-and-re-added key goes to the end. sections = the map as
   [{ key, text }] in order.
7. Offered tools. Replay the system messages of messages in order into an ordered map name -> declaration: for each
   message, first delete every name in message.toolsRemoved, then set each declaration of message.toolsAdded by its
   name (present: replaced in its position; new: at the end). tools = the map's declarations in order.

Return { head, entries: active, contributions, messages, sections, tools }.
