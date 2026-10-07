# pi-durable's two libraries: the exact surface it uses, and the natlang services that replace them

Source: `/home/werg/src/pi` at `f10993b`. Scope: every static import of `@earendil-works/pi-ai` (and its `utils/*`
subpaths) and of `@earendil-works/chord` (and `/context`, `/delta`) in `packages/durable/src` and
`packages/coding-agent/src/experimental/durable`. Neither package has dynamic imports or re-exports of these libraries.
The durable tools (`durable/src/tools/{bash,read,write,edit}.ts`) import `Type`/`Static` from `typebox` directly; that is
the same TypeBox that pi-ai re-exports, so they are counted under pi-ai. The goal is to describe each library's surface
so the port can put it behind a natlang service or replace it with a natlang mechanism. The libraries themselves are
not ported.

In short:

- **pi-ai.** Durable calls exactly five `Models` members (`getModel`, `streamSimple`, `completeSimple`,
  `fetchDeferred`, `cancelDeferred`) and ten pure helpers. It reads about fifteen message fields. Everything else stays
  inside the library: providers, auth, catalogs, wire conversion, cost and transport. One service, `ai`, is enough
  (A.11).
- **Chord.** pi-durable uses Chord for four things: (1) `Context` as a cancellation carrier only (no `Context.value`
  key is ever read); (2) strict JSON types and `copyJson`; (3) `Draft` plus the delta tracker (`track`, `Op`, `apply*`)
  behind `tx.doc()`; (4) `replicatedState` for read-only observation. Facets, services/RPC and `MutableReplicatedState`
  are not used. The harness logic itself relies on two of these: cancellation, and editing a document inside a commit
  (`Draft`). Everything else is host infrastructure (B.6).

---

## Part A — pi-ai (`@earendil-works/pi-ai`)

### A.1 Import inventory

| module | types imported | values imported | what the module does with them |
|---|---|---|---|
| durable `harness/types.ts` | `AssistantMessage, CacheRetention, Message, Models, ModelThinkingLevel, Static, Tool, ToolCall, ToolResultMessage, Transport, TSchema, Usage, UserMessage` | — | Public harness types: `ModelRef`, `UserInput = UserMessage["content"]`, `ToolRegistration = Tool & {...}`, `ConversationStreamOptions`, the hook signatures, `ContextView.messages`, and `models: Models` on `ToolExecutionApi`/`HookApi`/`HarnessOptions` |
| durable `types.ts` | `Message, Models` | — | `EntryRecord.model: Message[]`, `ContextEdit.messages`, `TaskRuntime.models` |
| durable `harness/generation.ts` | `Api, AssistantMessage, DeferredHandle, Message, Model, ModelThinkingLevel, SimpleStreamOptions, ToolCall` | `isContextOverflow` (`utils/overflow`), `isRetryableAssistantError`, `retryDelayMs` (`utils/retry`), `getCurrentTools` (`utils/transcript`) | Runs one model turn: request, stream, poll, classify; offered-tool check for the tool round |
| durable `harness/compaction.ts` | `AssistantMessage, Message, ModelThinkingLevel, SimpleStreamOptions` | `calculateContextTokens`, `estimateMessageTokens` (`utils/estimate`), `isRetryableAssistantError`, `retryDelayMs` | Threshold estimate, cut selection, summarization request and retry |
| durable `harness/prompt.ts` | `Message, SystemMessage, Tool, ToolReference` | `declarationsEqual`, `getCurrentTools`, `toToolDeclaration` (`utils/transcript`) | Plans the `pi.system` entries: section patches and tool loadout changes |
| durable `harness/context.ts` | `AssistantMessage, Message, ToolCall, ToolResultMessage` | — | Model-context derivation: excluded stop reasons, tool result ordering, synthesized missing results |
| durable `harness/tool.ts` | `ImageContent, TextContent, ToolCall, ToolResultMessage` | `validateToolArguments` (`utils/validation`) | Tool task: validation, building the result message |
| durable `harness/provider.ts` | — | `uuidv7` (`utils/uuid`) | `pi.provider.sessionId` initial value |
| durable `harness/usage.ts` | `Usage` | — | `pi.usage` ledger and `addUsage` |
| durable `harness/live.ts` | `AssistantMessage` | — | Type of the committed partial `pi.live.generation.message` |
| durable `harness/events.ts` | `AssistantMessage, Message, Usage` | — | Experimental UI event types (`message_update.usage`, `MessageChange.block`) |
| durable `harness/define.ts` | `TSchema` | — | Generic bound of `defineTool` |
| durable `harness/scheduler.ts` | `Models` | — | Passes `models` into every `TaskRuntime` |
| coding-agent `experimental/durable/runtime.ts` | `ModelThinkingLevel` | `clampThinkingLevel`, `getSupportedThinkingLevels` | `/model` and thinking-level cycling in the frontend |
| coding-agent `experimental/durable/harness-setup.ts` | `ModelThinkingLevel` | — | `InitialModel.thinkingLevel` |
| coding-agent `experimental/durable/subagent.ts` | `AssistantMessage` | `Type` (TypeBox builder) | Subagent tool schema `Type.Object({ task: Type.String({ description }) })`; reads the child's answer text |
| coding-agent `experimental/durable/tui.ts` | `AssistantMessage, ToolResultMessage, Usage, UserMessage` | — | Rendering only: message blocks, footer token and cost totals |

The frontend passes a `ModelRuntime` (coding-agent `core/model-runtime.ts`, `class ModelRuntime implements Models`) as
`HarnessOptions.models`. It is pi-ai's `Models` contract plus credentials, `models.json` and catalog refresh.

Never used by pi-durable: `Models.stream`/`complete` (the per-API option path), `getAuth`/`checkAuth`/`login`/`logout`/
`refresh`/`getAvailable`, `generateImages`, `classify`, `calculateCost` (providers apply it), `normalizeContext`
(`Models` applies it), `retryAssistantCall`, `estimateContextTokens`, `toolChoice`, `temperature`, `thinkingBudgets`,
`samplingParams`.

### A.2 Models and model lookup

| unit | signature | meaning | used by | port |
|---|---|---|---|---|
| `Models` | interface | The runtime collection of providers. It resolves auth per request and delegates to the provider that owns the model. Durable treats it as opaque: generation and compaction call it, and tools and hooks receive it as `api.models`. | `HarnessOptions.models` → `TaskRuntime.models`, `ToolExecutionApi.models`, `HookApi.models` | service `ai` |
| `Models.getModel` | `(provider: string, id: string) => Model<Api> \| undefined` | Synchronous lookup in the last-known chat catalog. | generation `prepare`/`request`/`poll`/abort; compaction `select`/`summarize`; frontend `setModel`/`cycleThinking` | `ai.model(ref)`. `undefined` fails the task `no_model` with `"No model is configured"` (no ref) or `` `Model ${provider}/${modelId} is not available` `` |
| `Models.streamSimple` | `(model: Model<Api>, context: { systemPrompt?: string; messages: Message[]; tools?: Tool[] }, options?: SimpleStreamOptions & { transformHeaders? }) => AssistantMessageEventStream` | Folds `systemPrompt`/`tools` into a leading system message (durable passes only `messages`, already led by its system message), resolves auth, and dispatches to `provider.streamSimple`. **Never throws.** A setup failure becomes one `error` event whose message has `stopReason: "error"` (examples: unknown provider, `"Provider is not configured: <provider>"`, an OAuth refresh failure). | generation `request` (`streamResponse`) | `ai.turn` |
| `Models.completeSimple` | same arguments `=> Promise<AssistantMessage>` | `streamSimple(...).result()`: the terminal message only. | compaction `summarize` | `ai.turn` without partials |
| `Models.fetchDeferred` | `(model, handle: DeferredHandle, options?: { signal?, wait?: number, ... }) => Promise<AssistantMessage>` | One status check of a deferred response (`wait` defaults to 0). A still-pending response returns `stopReason: "deferred"` with a new `deferred` handle. A provider without deferred support yields an `error` message. | generation `poll` | `ai.poll` |
| `Models.cancelDeferred` | `(model, handle, options?) => Promise<void>` | Best-effort cancellation. **Can reject**, e.g. `ModelsError("provider", "Provider X does not support deferred responses")`. | generation abort handler, `poll` phase only; a rejection is reported, not fatal | `ai.cancel` |
| `Model<Api>` | `{ id, name, api, provider, baseUrl, input: ("text"\|"image")[], inputLimits?, cost: ModelCost, headers?, type?: "chat", reasoning: boolean, thinkingLevelMap?, promptCache?, contextWindow, maxTokens, samplingParams?, samplingParamsByThinkingLevel?, compat? }` | Catalog entry. **Durable reads `contextWindow`** (compaction thresholds; `<= 0` disables them) **and `maxTokens`** (summary cap). The frontend reads `provider`, `id`, `name`, `contextWindow` and `reasoning`; `thinkingLevelMap` is read through the helpers. Every other field is read only by the library. | — | `ai.ModelInfo` keeps the read fields |
| `Api` | `KnownApi \| string` (`"anthropic-messages"`, `"openai-responses"`, `"openai-completions"`, `"google-generative-ai"`, `"bedrock-converse-stream"`, …) | Wire-protocol id of a model. Appears only as the type parameter of `Model<Api>`. | generation type | stays in library |
| `ModelRef` (durable's own) | `{ provider: string; modelId: string }` | The durable identity of a model. Stored in `pi.agent`, in generation `request`/`poll` checkpoints and in compaction `SummaryRequest`, and resolved at each use. | everywhere | typed JSON value |
| `getSupportedThinkingLevels` | `(model: Model) => ModelThinkingLevel[]` | `reasoning` false gives `["off"]`. Otherwise `["off","minimal","low","medium","high","xhigh","max"]`, keeping a level unless `thinkingLevelMap[level] === null`. `xhigh` and `max` are kept only when the map has an entry for them (not undefined). | frontend `cycleThinking` (next level = following entry, wrapping) | `ai.ModelInfo.thinkingLevels` |
| `clampThinkingLevel` | `(model, level) => ModelThinkingLevel` | A supported level is returned unchanged. Otherwise take the first supported level above it in the order off < minimal < low < medium < high < xhigh < max, else the nearest supported level below, else the first supported level, else `"off"`. | frontend `setModel` keeps the conversation's level across a model change | inline rule in the frontend's configure step, over `thinkingLevels` |
| `ModelRuntime` (coding-agent, not a pi-ai import) | `ModelRuntime.create(options)`, `getAvailableSnapshot(): Model[]`, `getModel`, plus `Models` | Loads the credential store (`auth.json`), `models.json`, the built-in and remote catalogs, and runs an availability refresh. `getAvailableSnapshot()` lists the models whose provider has auth. | frontend `openDurable`, model picker | host constructs the `ai` service from it |

### A.3 Providers and auth configuration

| concern | where it happens | durable's involvement | port |
|---|---|---|---|
| Provider for a model | `Models` picks `provider.id === model.provider` (`requireChatProvider`) | none | inside `ai` |
| Auth resolution | `Models.applyAuth`: `getAuth(model)` gives `{ apiKey, headers, env, baseUrl }`. Explicit options win per field, `transformHeaders` runs last, and `auth.baseUrl` overrides `model.baseUrl`. No auth: `ModelsError("auth", "Provider is not configured: X")`, delivered as an error message. | none | inside `ai` |
| Credentials, OAuth, login/logout, credential store | `CredentialStore`, `AuthContext`, OAuth flows | none (the frontend gets them through `ModelRuntime.create()`) | host configuration, **out** of the harness |
| Catalogs and refresh | `getModels`, `refresh`, `ModelsStore`, `filterModels` | none | host |
| HTTP proxy and idle timeouts | coding-agent `configureHarnessHttp` (undici dispatcher), not pi-ai | frontend only | host |
| Provider retries inside one request | `StreamOptions.maxRetries`, `maxRetryDelayMs` (SDK level) | passed through from `ConversationStreamOptions` | `ai.turn` option |

### A.4 Request options (`SimpleStreamOptions` as durable fills them)

Generation's request is `{ ...settings.stream, signal: runtime.signal, sessionId, reasoning? }`. Compaction's request is
`{ ...settings.stream without deferred, cacheRetention: "none", maxTokens, signal, sessionId, reasoning? }`, where
`maxTokens = min(floor(0.8 × reserveTokens), model.maxTokens if > 0 else ∞)`. The `stream` fields form durable's
curated `ConversationStreamOptions`. The generation `request` checkpoint pins them, so a resend after recovery uses
the same values.

| field | type | meaning | durable source |
|---|---|---|---|
| `signal` | `AbortSignal` | Aborting it ends the stream with `stopReason: "aborted"` | `runtime.signal`, the task invocation's own AbortController |
| `sessionId` | `string` | Provider-side session identity, used for cache routing and affinity. Providers that do not support it ignore it. | `pi.provider.sessionId` (`ensureProviderSessionId`) |
| `reasoning` | `ThinkingLevel` | Requested thinking effort, mapped per model through `thinkingLevelMap` | `agent.thinkingLevel`, **omitted when `"off"`** |
| `maxTokens` | `number` | Output cap | compaction only |
| `transport` | `"sse" \| "websocket" \| "websocket-cached" \| "auto"` | Preferred transport where a provider has several | settings |
| `timeoutMs` | `number` | HTTP request timeout (SDK default 10 min) | settings. The frontend sets the provider timeout, else the idle timeout, with 0 mapped to 2147483647 |
| `maxRetries` | `number` | SDK-level retries inside one attempt (SDK default 2) | settings |
| `maxRetryDelayMs` | `number` | Above this server-requested delay the request fails instead of waiting (default 60000, 0 = no cap) | settings |
| `headers` | `Record<string,string>` | Extra HTTP headers merged over provider defaults | settings |
| `metadata` | `JsonObject` | Request metadata (e.g. Anthropic `user_id`) | settings |
| `cacheRetention` | `"none" \| "short" \| "long"` | Prompt-cache lifetime preference (default `"short"`) | settings; compaction forces `"none"` |
| `deferred` | `boolean \| { window?: "15m"\|"1h"\|"24h" }` | Ask a capable provider for an asynchronous (batch) response and a handle | settings; never sent by compaction |

### A.5 Streaming calls and their events

| unit | signature | meaning | how durable consumes it |
|---|---|---|---|
| `AssistantMessageEventStream` | `AsyncIterable<AssistantMessageEvent> & { result(): Promise<AssistantMessage> }` | One response. `result()` resolves with the terminal message (`done.message` or `error.error`). The stream stamps `durationMs` on the final message. | `streamResponse` iterates it and then awaits `result()` |
| `start` | `{ type, partial }` | Opening event; `partial.content` is empty | skipped (empty content) |
| `text_start` / `text_delta{delta}` / `text_end{content}` | `+ contentIndex, partial` | A text block opens, grows and closes | non-terminal: the `partial` is taken |
| `thinking_start` / `thinking_delta` / `thinking_end` | same | A thinking block; a redacted one may be complete at start | same |
| `toolcall_start` / `toolcall_delta{delta}` / `toolcall_end{toolCall}` | same | A tool call; `delta` is JSON text of its arguments | same |
| `done` | `{ reason: "stop"\|"length"\|"toolUse"\|"deferred", message }` | Terminal success | skipped in the loop; `result()` returns it |
| `error` | `{ reason: "aborted"\|"error", error: AssistantMessage }` | Terminal failure, including setup failure | skipped in the loop; `result()` returns it |
| `partial` | `AssistantMessage` | **The provider's shared, mutable response-so-far**, not a snapshot | copied synchronously (`copyJson(partial, { omitUndefinedProperties: true })`) before an async commit |

Partial throttle in `streamResponse`, which is **observation, host-side**:
- Keep the newest non-terminal partial with non-empty content as `pending`.
- If no timer is set and no commit is in flight, set a timer for `settings.progress.partialIntervalMs` (default 100 ms).
- When the timer fires, copy `pending` and commit it as `pi.live.generation.message`, written leaf by leaf with
  `assignJson` so Chord records string appends. Set `live.generation ??= { attempt }` first.
- A commit rejection is reported unless the signal is aborted.
- After the commit, re-arm the timer if a newer partial is waiting.
- In `finally`, stop the throttle, clear the timer and await the in-flight commit, so no stale partial lands after the
  outcome.
- A deferred response never has content, so it never leaves a partial.

### A.6 Messages and content blocks

`Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage`. Messages are stored verbatim in
`EntryRecord.model` and sent back verbatim in later requests. **Opaque provider fields (signatures, ids) must survive
unchanged.**

| type | field | type | meaning | durable reads / writes |
|---|---|---|---|---|
| `SystemMessage` | `role` | `"system"` | | writes |
| | `content` | `string \| TextContent[]` | Base prompt on the leading message, additional instructions on later ones | **always writes `""`** in `pi.system` entries; the summarizer request uses `SUMMARIZATION_SYSTEM_PROMPT` |
| | `sections?` | `Record<string, string \| null>` | Named ordered prompt sections. A string adds or replaces a section, `null` removes it. Replaying all system messages in order gives the current prompt. | writes the planned patches; reads them in `replaySections` |
| | `toolsAdded?` | `Tool[]` | Tool definitions that become available here | writes `toToolDeclaration` results |
| | `toolsRemoved?` | `ToolReference[]` | Tools removed here; applied before `toolsAdded` within one message | writes |
| | `timestamp` | `number` (ms) | | `runtime.now()` |
| `UserMessage` | `role`, `content: string \| (TextContent \| ImageContent)[]`, `timestamp` | | User input (`UserInput` is its `content`). Also used for the compaction summary text, a reset handoff and an `onYield` continuation. | writes; the summary text is `"The conversation history before this point was compacted into the following summary:\n\n<summary>\n" + summary + "\n</summary>"` |
| `AssistantMessage` | `role` | `"assistant"` | | |
| | `content` | `(TextContent \| ThinkingContent \| ToolCall)[]` | Ordered blocks of the answer | **reads**: tool calls for the round, text for summaries and subagent answers |
| | `api`, `provider`, `model` | `string` | Which API, provider and model id answered | **reads `provider` and `model`** for the usage key `` `${provider}/${model}` `` |
| | `responseModel?`, `responseId?`, `providerThinkingLevel?`, `thinkingLevel?`, `diagnostics?`, `rawStopReason?`, `endTurn?` | | Provider reporting, debugging only | opaque, stored |
| | `usage` | `Usage` | Tokens and cost of this response | **reads** (ledger, threshold estimate) |
| | `stopReason` | `StopReason` | How the response ended | **reads** (classification, context exclusion) |
| | `deferred?` | `DeferredHandle` | Present when `stopReason === "deferred"` | **reads** |
| | `errorMessage?` | `string` | Provider or runtime error text | **reads** (classification, failure detail) |
| | `timestamp`, `durationMs?` | `number` | Request start; response duration | opaque |
| `ToolResultMessage` | `role: "toolResult"`, `toolCallId`, `toolName` | | Pairs the result with its call | writes |
| | `content` | `(TextContent \| ImageContent)[]` | What the model sees | writes the result content, followed by a text block `"<harness>\n" + lines "[severity] message" + "\n</harness>"` when there are diagnostics |
| | `details?` | JSON | Structured data for UIs, not sent to the model | writes when defined |
| | `usage?` | `Usage` | Spend of the tool itself (e.g. a nested model call) | writes when defined; also added to `pi.usage.tools[toolName]` |
| | `nestedCalls?` | | Calls the tool made to other tools | not written |
| | `isError` | `boolean` | | writes (default `false`) |
| | `timestamp`, `durationMs?` | | | writes; `durationMs` is the measured execution time |
| `TextContent` | `{ type: "text"; text; textSignature? }` | | Text. `textSignature` is OpenAI Responses metadata. | reads `text` |
| `ThinkingContent` | `{ type: "thinking"; thinking; thinkingSignature?; redacted? }` | | Reasoning text plus an opaque replay signature (redacted: payload in the signature) | reads `thinking` (compaction serialization) |
| `ImageContent` | `{ type: "image"; data: base64; mimeType }` | | Image input or result | passes through |
| `ToolCall` | `{ type: "toolCall"; id; name; arguments: JsonObject; thoughtSignature?; namespace? }` | | A call the model made | reads `id`, `name`, `arguments` |

Synthesized messages durable creates itself:
- A missing tool result: `{ role: "toolResult", toolCallId, toolName, content: [{ type: "text", text: "Tool result
  unavailable: history ends before this call completed." }], isError: true, details: { reason: "missing_result" },
  timestamp: <assistant's> }`.
- A converted partial: the committed partial with `stopReason: "aborted"`.

### A.7 Tools and schemas

| unit | signature | meaning | durable use |
|---|---|---|---|
| `Tool<TParameters>` | `{ name; description; parameters: TSchema; constrainedSampling?: false \| { type: "json_schema"; strict: "prefer"\|"require" } \| { type: "grammar"; variants: { openai_lark?; openai_regex? } } }` | The model-facing declaration | `ToolRegistration = Tool & { replay?, executionMode?, prepareArguments?, outputLimits?, execute }`. Only the `Tool` fields enter the transcript. |
| `ToolReference` | `{ name }` | Names a tool in `toolsRemoved` | written by `planTools` |
| `TSchema` / `Static<T>` | TypeBox schema type / its static TypeScript type | Parameter schema; `Static` types `execute`'s `args` | type-level only |
| `Type` | TypeBox builder | Builds JSON-Schema objects carrying TypeBox symbol keys | subagent and durable tools build their `parameters` with it |

In natlang, a tool's `parameters` is a plain JSON-Schema value (typed JSON). Validation is a service call (A.10).

### A.8 Usage and cost

| field | meaning |
|---|---|
| `input`, `output`, `cacheRead`, `cacheWrite` | Token counts. `output` includes reasoning tokens. |
| `cacheWrite1h?` | The part of `cacheWrite` written with 1 h retention (Anthropic only) |
| `reasoning?` | Reasoning tokens, a subset of `output`, when the provider reports them |
| `totalTokens` | Provider total (may be 0) |
| `cost: { input, output, cacheRead, cacheWrite, total }` | USD. **Computed inside the library** (`calculateCost`). Rates are $ per million tokens. The tier with the highest `inputTokensAbove` below `input + cacheRead + cacheWrite` applies to the whole request. 1 h cache writes are charged at 2 × the base input rate. |

Durable's use:
- `recordUsage` adds a response's usage to `pi.usage.models["provider/model"]`, or a tool result's usage to
  `pi.usage.tools[toolName]`, in the commit that appends the entry.
- Every summarization attempt is recorded too, including failed and aborted ones.
- `addUsage` sums every counter. Optional counters are added once either side reports them.
- The frontend footer sums `input`, `output`, `cacheRead`, `cacheWrite` and `cost.total`. Its context size is
  `totalTokens || input + output + cacheRead + cacheWrite` of the newest assistant message that is not
  `aborted`/`error` and comes after the newest compaction.

### A.9 Enumerations and handles

| unit | values | meaning | durable classification |
|---|---|---|---|
| `StopReason` | `"pending"` | Initial value of a provider's partial (Google, Azure adapters) | never classified; a committed partial may carry it, and `convertPartial` rewrites it to `"aborted"` |
| | `"stop"` | Final answer | answer: `onYield`, then the final boundary |
| | `"length"` | Hit the output cap | generation: answer (like `stop`). Compaction: failure `"Summarization hit the token limit; the summary is incomplete"`. |
| | `"toolUse"` | Ended to call tools | with ≥ 1 `toolCall`: tool round; without: answer |
| | `"error"` | Failed; see `errorMessage` | overflow → compaction; retryable → retry; otherwise fail with `model_error` |
| | `"aborted"` | Cancelled | Checked first with `runtime.signal.throwIfAborted()`: under an abort mark, nothing is classified and the abort phase handles it. Otherwise it fails with `model_error`. |
| | `"deferred"` | Accepted asynchronously; `deferred` holds the handle | move to `poll` |
| | (context derivation) | | entries with `aborted`, `error` or `deferred` are **excluded** from future requests |
| `CacheRetention` | `"none" \| "short" \| "long"` | Prompt-cache lifetime; providers map it to their own values | setting; compaction uses `"none"` |
| `ModelThinkingLevel` | `"off" \| ThinkingLevel`; `ThinkingLevel = "minimal"\|"low"\|"medium"\|"high"\|"xhigh"\|"max"` | Requested reasoning effort | `"off"` omits `reasoning`. Stored in `pi.agent` and the request checkpoint. |
| `Transport` | `"sse" \| "websocket" \| "websocket-cached" \| "auto"` | Preferred streaming transport | setting pass-through |
| `DeferredHandle` | `{ provider; modelId; api; id; expiresAt?; pollAfterMs?; data?: JsonValue }` | Provider token for an asynchronous response, plus the conversion data needed to rebuild the message | Stored **verbatim** in the `poll` checkpoint. Durable reads only `pollAfterMs`: `pollAt = max(now + (pollAfterMs ?? 5000), previousPollAt + 1)`. |

### A.10 Pure helper functions

| function | signature | exact rule | callers | port |
|---|---|---|---|---|
| `isContextOverflow` | `(message, contextWindow?) => boolean` | Durable passes no `contextWindow`, so only this case applies: `stopReason === "error"`, the `errorMessage` matches one of 25 provider patterns (e.g. `/prompt (?:is )?too long/i`, `/exceeds the context window/i`, `/maximum context length is \d+ tokens/i`, `/context[_ ]length[_ ]exceeded/i`, `/too many tokens/i`, …) and none of `/^(Throttling error\|Service unavailable):/i`, `/rate limit/i`, `/too many requests/i`. Also true for provider `cerebras` with `/^4(?:00\|13)\s*(?:status code)?\s*\(no body\)/i`. | generation classify (`overflow = error && isContextOverflow`) | `ai.failure(message).overflow`: the pattern table is provider knowledge that stays in the library |
| `isRetryableAssistantError` | `(message) => boolean` | False unless `stopReason === "error"` with a non-empty `errorMessage`. False if it matches a quota or billing pattern (`GoUsageLimitError`, `FreeUsageLimitError`, `Monthly usage limit reached`, `available balance`, `insufficient_quota`, `out of budget`, `quota exceeded`, `billing`, `subscription_sharing_usage_limit_exceeded`). Otherwise true if it matches a transient pattern (`overloaded`, `rate.?limit`, `429`, `5xx`, `network/connection` errors, `timeout`, `terminated`, `websocket closed`, `stream ended before…`, `retry delay`, `you can retry your request`, `ResourceExhausted`, …). | generation (only when not overflow); compaction (no overflow check) | `ai.failure(message).retryable` |
| `retryDelayMs` | `(policy: { baseDelayMs; maxAgentDelayMs? }, attempt) => number` | `min(baseDelayMs × 2^max(0, attempt−1), maxAgentDelayMs ?? 60000)`. Durable's default policy is `{ enabled: true, maxRetries: 3, baseDelayMs: 2000, maxAgentDelayMs: 60000 }`, and a retry is allowed while `attempt <= maxRetries`. | generation, compaction | **inline** arithmetic in the retry instruction (or eval) |
| `getCurrentTools` | `(messages) => Tool[]` | Replay the system messages in order. For each, delete `toolsRemoved` by name, then set `toolsAdded` by name. A name re-added after removal moves to the end; a re-set without removal keeps its position. | prompt planning; generation tool round (`offered` names) | crisp helper `transcript.currentTools` (or `ai.*` if folders cannot share it) |
| `toToolDeclaration` | `(tool) => Tool` | `{ name, description, parameters: JSON round-trip, constrainedSampling if defined }`, which drops implementation fields and TypeBox symbols | prompt planning | crisp (same helper) |
| `declarationsEqual` | `(a, b) => boolean` | `JSON.stringify(toToolDeclaration(a)) === JSON.stringify(toToolDeclaration(b))` | prompt planning | crisp (same helper) |
| `calculateContextTokens` | `(usage) => number` | `totalTokens \|\| input + output + cacheRead + cacheWrite` | compaction `estimateContext` | inline or crisp |
| `estimateMessageTokens` | `(message) => number` | `ceil(chars / 3.5)`. User and tool-result messages count text length, 4800 chars per image. Assistant messages count text + thinking + `name` + `JSON.stringify(arguments)`. A system message counts its text (content, then the non-null section texts, non-empty parts joined with `"\n\n"`) plus `ceil(JSON.stringify(toolsAdded).length / 3.5)` + the same for `toolsRemoved`. | compaction `selectCut`, `estimateContext`, generation threshold check | `ai.estimateTokens(messages): number[]` (pi's heuristic; or crisp) |
| `validateToolArguments` | `(tool, call) => args` (throws) | 1. Clone the arguments. 2. Delete a property set to `null` when it is optional, its schema has no `$ref` and the schema rejects `null`. 3. TypeBox `Value.Convert`. 4. For a plain-JSON-Schema tool (no TypeBox kind symbol), coerce primitives recursively: number/integer from a numeric string, a boolean or `null`→0; boolean from `"true"`/`"false"`, 1/0 or `null`→false; string from a number, a boolean or `null`→`""`. Union members (`anyOf`/`oneOf`) are tried in order. `allOf`, nested properties, `additionalProperties` and items are coerced too. 5. Check. On failure throw `` `Validation failed for tool "${name}":\n` `` followed by one line `"  - <path>: <message>"` per error (path dot-joined, `"root"` when empty, a `required` error names the property), then `"\n\nReceived arguments:\n" + JSON.stringify(original, null, 2)`. | tool task (twice: before and after `beforeTool`) | `ai.validateArguments`: library-owned behavior, and it must match the schema dialect |
| `uuidv7` | `(timestampMs?) => string` | Time-ordered UUID v7 | `pi.provider` initial `sessionId` | `ai.newSessionId()` (randomness and clock live in the host) |

### A.11 Proposed natlang service `ai`

This is the declaration the model reads (`serviceDeclarations.ai`). It runs one model turn and reads its result. The
type declarations are part of the service's `.d.ts`, so harness functions can type their arguments and results with
them. Cancellation is the calling task's `AbortSignal`, which the host binds. No `Context` parameter is exposed.

```ts
export type ModelRef = { provider: string; modelId: string };
export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

/** What harness rules need to know about a model. */
export type ModelInfo = {
  provider: string; modelId: string; name: string;
  /** Tokens one request may hold; 0 when unknown (no compaction thresholds then). */
  contextWindow: number;
  /** Longest answer the model can produce; 0 when unknown. */
  maxTokens: number;
  /** False: only "off" exists. */
  reasoning: boolean;
  /** Supported levels in the order off, minimal, low, medium, high, xhigh, max. */
  thinkingLevels: ThinkingLevel[];
  input: ("text" | "image")[];
};

export type TextContent = { type: "text"; text: string; textSignature?: string };
export type ThinkingContent = { type: "thinking"; thinking: string; thinkingSignature?: string; redacted?: boolean };
export type ImageContent = { type: "image"; data: string; mimeType: string };
export type ToolCall = { type: "toolCall"; id: string; name: string; arguments: JsonObject; thoughtSignature?: string; namespace?: string };
export type Tool = { name: string; description: string; parameters: JsonObject; constrainedSampling?: JsonValue };
export type Usage = { input: number; output: number; cacheRead: number; cacheWrite: number; cacheWrite1h?: number;
  reasoning?: number; totalTokens: number; cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number } };
export type DeferredHandle = { provider: string; modelId: string; api: string; id: string; expiresAt?: number; pollAfterMs?: number; data?: JsonValue };
export type SystemMessage = { role: "system"; content: string | TextContent[]; sections?: Record<string, string | null>;
  toolsAdded?: Tool[]; toolsRemoved?: { name: string }[]; timestamp: number };
export type UserMessage = { role: "user"; content: string | (TextContent | ImageContent)[]; timestamp: number };
/**
 * Store it exactly as returned. It also carries provider fields not listed here (responseId, diagnostics,
 * durationMs, …) and the signatures in its blocks; they are needed to send it back later.
 */
export type AssistantMessage = { role: "assistant"; content: (TextContent | ThinkingContent | ToolCall)[];
  api: string; provider: string; model: string; usage: Usage;
  stopReason: "stop" | "length" | "toolUse" | "error" | "aborted" | "deferred";
  errorMessage?: string; deferred?: DeferredHandle; timestamp: number };
export type ToolResultMessage = { role: "toolResult"; toolCallId: string; toolName: string;
  content: (TextContent | ImageContent)[]; details?: JsonValue; usage?: Usage; isError: boolean; timestamp: number; durationMs?: number };
export type Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage;

export type TurnOptions = {
  /** "off" requests no reasoning. */
  thinkingLevel: ThinkingLevel;
  /** The conversation's provider identity (pi.provider.sessionId). */
  sessionId: string;
  maxTokens?: number;
  cacheRetention?: "none" | "short" | "long";
  transport?: "sse" | "websocket" | "websocket-cached" | "auto";
  timeoutMs?: number; maxRetries?: number; maxRetryDelayMs?: number;
  headers?: Record<string, string>; metadata?: JsonObject;
  deferred?: boolean | { window?: "15m" | "1h" | "24h" };
};

/** null: the model is not in the catalog. */
export function model(ref: ModelRef): ModelInfo | null;
/** Models whose provider has credentials, for choosing one. */
export function available(): ModelInfo[];
/**
 * Run one model turn over `messages` (led by its system message) and return the finished message.
 * Never throws for provider trouble: a failure is stopReason "error" with errorMessage, and cancelling the task
 * gives "aborted". With `live`, the host publishes the growing answer as that conversation's live partial and
 * finishes publishing before this returns.
 */
export function turn(model: ModelRef, messages: Message[], options: TurnOptions,
  live?: { conversationId: number; attempt: number }): Promise<AssistantMessage>;
/** Check a deferred response once; still pending: stopReason "deferred" with a new handle. */
export function poll(model: ModelRef, handle: DeferredHandle): Promise<AssistantMessage>;
/** Best-effort cancel of a deferred response; may reject. */
export function cancel(model: ModelRef, handle: DeferredHandle): Promise<void>;
/** overflow: the request was too long for the model. retryable: a transient provider or network failure. */
export function failure(message: AssistantMessage): { overflow: boolean; retryable: boolean };
/** pi's per-message token estimate (characters / 3.5, images 4800 characters). */
export function estimateTokens(messages: Message[]): number[];
/** Validate and coerce tool arguments against a JSON Schema the way pi does; error is pi's exact message. */
export function validateArguments(toolName: string, parameters: JsonObject, args: JsonObject):
  { args: JsonObject } | { error: string };
/** A fresh provider session identity (UUID v7). */
export function newSessionId(): string;
```

| `ai` member | pi-ai underneath | who calls it in the port |
|---|---|---|
| `model` | `Models.getModel` + `getSupportedThinkingLevels` | generation prepare/request/poll, compaction select/summarize, frontend |
| `available` | `ModelRuntime.getAvailableSnapshot` | frontend model picker |
| `turn` | `Models.streamSimple` with the partial throttle (A.5), or `completeSimple` when `live` is absent | generation request (with `live`), compaction summarize (without), tools via `api.models` |
| `poll` / `cancel` | `fetchDeferred` / `cancelDeferred` | generation poll / abort |
| `failure` | `isContextOverflow`, `isRetryableAssistantError` | generation and compaction classification. Two booleans, because compaction checks only `retryable` and generation checks `overflow` first. |
| `estimateTokens` | `estimateMessageTokens` | compaction cut and threshold |
| `validateArguments` | `validateToolArguments` | tool task |
| `newSessionId` | `uuidv7` | conversation creation |

With this surface, a natural-language generation function needs no pi-ai knowledge beyond the declared types. It
resolves the model, sends the committed context, and reads `stopReason`, `content` (tool calls, text), `errorMessage`,
`usage`, `provider`/`model` and `deferred`. It then classifies with `failure()` and the rules of spec §8.3.

**Stays inside the library (behind `ai.turn`):**
- provider HTTP/WebSocket transports and SDKs, and SDK-level retries;
- auth resolution, the credential store, OAuth refresh, header merging and the `baseUrl` override;
- model catalogs and refresh;
- `normalizeContext` and per-API transcript conversion: collapsing system messages for models without mid-conversation
  system messages, anchoring tool additions, cache-control markers, and thinking-level mapping through
  `thinkingLevelMap`/sampling parameters;
- partial-JSON parsing of tool-call deltas;
- cost calculation;
- `durationMs` timing;
- error normalization (`lazyStream`);
- the encoding and decoding of deferred handles;
- image resize and input limits;
- the overflow and retry pattern tables.

**Crisp in the port:** the transcript helpers `currentTools`, `toToolDeclaration` and `declarationsEqual`, if the
generation and prompt-planning functions share a folder; otherwise these are `ai` members too. `retryDelayMs` and
`calculateContextTokens` are inline arithmetic.

---

## Part B — Chord (`@earendil-works/chord`, `/context`, `/delta`)

### B.1 Import inventory

| module | `chord` | `chord/context` | `chord/delta` | role |
|---|---|---|---|---|
| durable `types.ts` | `AttachedReplicatedState, Context, Draft, JsonValue` | — | `Op` | Public contracts: `Tx.doc()` returns `Draft<T>`, `DocumentState`, `WatchHandle.start(listener(value, ops, context))`, `checkpointWhen(value, ops, info)`, `CommitPublication`, `DocumentContent` deltas |
| durable `entries.ts`, `harness/define.ts`, `harness/json.ts`, `testing/storage-benchmark.ts` | `JsonValue` | (`BACKGROUND_CONTEXT` in benchmark) | — | JSON typing (`assignJson` leaf-by-leaf writer) |
| durable `documents.ts` | `copyJson, JsonValue` | — | `Op` | Document definitions; migration result copied to strict JSON |
| durable `session/session.ts` | `Context, JsonValue, replicatedState` | `awaitWithContext, withoutAbortSignal` | `Op, track` | Session line: commit, cache of tracked documents, `documentState()`, close |
| durable `session/transaction.ts` | `Context, copyJson, Draft, JsonValue` | — | `Change, Op, Prepared, Tracker, track` | One commit: draft acquisition, prepare/adopt/abort, base-or-delta choice |
| durable `session/observation.ts` | `Context, JsonValue, ReplicatedStateSource, ReplicatedStateSourceAttachment, ReplicatedStateSourceFrame` | `withoutAbortSignal` | `Op` | `CommittedStateSource` (Session → Chord bridge), `CommittedWatch` (bounded exact-frame watch) |
| durable `session/forks.ts`, `env/index.ts`, `env/node.ts`, `env/node-watch.ts`, `tools/{read,bash,path-utils,file-mutation-queue}.ts`, `storage/jsonl/node.ts`, `harness/util.ts`, `harness/prompt.ts`, `harness/context.ts`, `harness/provider.ts`, `harness/compaction.ts` | `Context` (compaction also `Draft`) | — | — | Cancellation parameter |
| durable `storage/memory.ts` | `Context, JsonValue` | — | `applyImmutableBatches, Op` | Replays stored deltas |
| durable `storage/sqlite/storage.ts` | `Context, JsonValue` | — | `apply, Op` | Replays stored deltas (mutable) |
| durable `storage/jsonl/storage.ts` | `Context, JsonValue` | — | — | |
| durable `harness/harness.ts` | `AttachedReplicatedState, Context, JsonValue` | `withAbortSignal, withoutAbortSignal` | — | `viewState()`, `taskGraph()`, bound conversation handles, open/close |
| durable `harness/scheduler.ts` | `Context, copyJson, JsonValue` | `awaitWithContext, withAbortSignal` | — | Per-invocation contexts; migration copies |
| durable `harness/view.ts` | `AttachedReplicatedState, Context, JsonValue, replicatedState` | `withoutAbortSignal` | `applyImmutable, NonEmptyPath, Op, Path` | Conversation view mount (spec §9.3) |
| durable `harness/task-graph.ts` | `AttachedReplicatedState, Context, JsonValue, replicatedState` | `withoutAbortSignal` | `applyImmutable, Op` | Task graph mount (spec §9.5) |
| durable `harness/events.ts` | `Context, JsonValue` | — | `Op, Path` | Translates view op batches into `AgentEvent`s (spec §9.4) |
| durable `harness/generation.ts` | `Context, copyJson, Draft, JsonValue` | — | — | Partial snapshot copy; `Draft<LiveState>` |
| durable `harness/tool.ts` | `Context, copyJson, JsonValue` | `awaitWithContext` | `overlap` | Strict-JSON copies of tool reports; cancellable wait for a details commit; progress byte estimate |
| durable `harness/agent.ts`, `harness/usage.ts`, `harness/inbox.ts`, `harness/live.ts`, `harness/submissions.ts` | `copyJson, Draft, JsonRepresentation` (subset each) | — | — | Document state types and draft edits |
| durable `testing/{env,storage}-conformance.ts` | `Context` | `BACKGROUND_CONTEXT, withAbortSignal` | `Op` | Tests (out) |
| coding-agent `experimental/durable/runtime.ts` | `AttachedReplicatedState` | `BACKGROUND_CONTEXT` | — | Frontend: subscribes to the conversation view and the task graph; every call uses the never-cancelled root context |
| coding-agent `experimental/durable/subagent.ts`, `harness-setup.ts` | `Context` | — | — | Cancellation parameter of `execute` / `cleanup` |

**Not used:** `defineFacet`, `createFacetHost`, `defineService`, the remote-service binding and transport, wire codecs,
`MutableReplicatedState`, the `replicatedState(initial)` overload, `withCancel`, `TODO_CONTEXT`, `createContextKey`,
`withContextValue`, `Context.value()` (verified: no keyed value is read), `isJsonValue`, `diffRevisions`, and the
`encoder`/`decoder` wire ops.

### B.2 Context and cancellation

| unit | signature | role | where | harness logic or host |
|---|---|---|---|---|
| `Context` | `interface { readonly abortSignal: AbortSignal \| undefined; value<T>(key): T \| undefined; toString() }` | Immutable invocation-scoped values. **pi-durable uses only `abortSignal`.** It is the trailing parameter of nearly every async API: `PhaseHandler(task, runtime, context)`, every `TaskRuntime` method, hooks `(…, api, context)`, `tool.execute(args, api, context)`, `PromptSection.render(input, context)`, `Submission.status/wait/abort`, and every `Conversation`/`Harness` and storage method. | everywhere | Both. The harness rule that depends on it is below. |
| `BACKGROUND_CONTEXT` | `const Context` | Root context with no signal | frontend `runtime.ts` (every call), tests | host / out |
| `withAbortSignal` | `(signal, context) => Context` | Child context cancelled by the parent's signal or by `signal` (`AbortSignal.any`) | scheduler: each task invocation gets `withAbortSignal(controller.signal, base)`. `TaskRuntime.waitForTask` runs under the invocation signal. Bound conversation handles given to tasks and tools bind the invocation signal to every call. | host (scheduler) |
| `withoutAbortSignal` | `(context) => Context` | Same values, no cancellation: "mandatory cleanup only" | (1) The Session's storage commit once admitted: caller cancellation must not interrupt settlement. (2) Close cleanup. (3) Commit-publication frame contexts: observers do not inherit the producer's cancellation. (4) The scheduler's base context. (5) `Harness.open` closing after a failed open. | host |
| `awaitWithContext` | `(promise, context) => Promise<T>` | Cancel *the wait*, not the work | `abortTask` joins the signalled run; tool `api.details()` waits for its progress commit (the update stays); `Session.close` waits for closing; `agent()` resolution in a task | host |
| `runtime.signal` (durable's own, raw `AbortSignal`) | `TaskRuntime.signal` | The invocation's AbortController: aborted by `abortTask`, Harness close or the end of the invocation | passed to pi-ai `signal`; checked by the harness rules below | harness logic reads it |

**The harness rule that depends on cancellation** (generation, tool, prompt, compaction): *an error raised after the
invocation is signalled propagates*. The invocation then ends without writing, and the separate abort phase settles
the state. Before that, an error becomes a result:
- a prompt section that throws is reported and keeps its shown text;
- a failure to build the environment is reported;
- a `beforeTool` throw blocks the call;
- an `execute` throw gives an `isError` result and ends the task `failed`;
- `classify`/`summarize` call `runtime.signal.throwIfAborted()` before their commit.

In natlang the task's `AbortSignal` (`runtime.run(fn, { signal })`) aborts the natlang calls in flight along the call
tree. A natural-language phase function therefore never handles a context. Phase functions need no cancellation
instruction: the host aborts the call and then runs the abort-phase function as a fresh call. The
`withoutAbortSignal`/`awaitWithContext` cases are all host plumbing.

### B.3 JSON values

| unit | signature | role | where | port |
|---|---|---|---|---|
| `JsonValue` | `null \| boolean \| number \| string \| JsonValue[] \| { [k]: JsonValue }` | Strict JSON. Durable defines `JsonObject` over it. | records, documents, task input/checkpoint/result, memos, tool details | natlang typed JSON values |
| `JsonRepresentation<T>` | type-level map of `T` to its strict-JSON shape | Types document fields that hold pi-ai values: `UsageState.models/tools: Record<string, JsonRepresentation<Usage>>`, `LiveState.generation.message`, `InboxItem.content` | type-level | typed JSON value types (no equivalent needed) |
| `copyJson` | `(value, { omitUndefinedProperties? }) => JsonValue` | An alias-free strict-JSON copy. It throws on cycles, non-finite numbers, symbols, class instances, sparse arrays and `undefined`, except that `omitUndefinedProperties` drops `undefined` object properties. | It (1) snapshots pi-ai's mutable partial before an async commit; (2) strips `undefined` from usage, tool details, diagnostics, `control` and submission content before they enter a draft, which rejects `undefined`; (3) copies migrated checkpoints and inputs and the owner's agent on conversation creation | **implicit**: natlang values are already JSON. The partial snapshot is host-side inside `ai.turn`. |

### B.4 Draft and delta tracking

| unit | signature | role | where | harness logic or host |
|---|---|---|---|---|
| `Draft<T>` | mutable mapped type of a JSON value (depth ≤ 8) | **The transaction-scoped mutable view of a document**, returned by `tx.doc(token, …)`. Harness code edits it in place: `live.generation = {...}`, `delete live.tools`, `live.tools = slots`, `items.push(...)`, `slot.status = "running"`, `totals[key] = …`. Rules: assigning `undefined` to a property deletes it; placed values are cloned and validated, and a non-JSON value throws before the draft changes; arrays stay dense; draft handles are unusable after the commit callback. | generation, compaction, tool, live, inbox, submissions, agent, usage | **Harness logic relies on it.** It is how a commit reads and edits `pi.live`, `pi.inbox`, `pi.agent`, `pi.usage` and `pi.provider` together with table writes, atomically. |
| `track` / `Tracker<T>` | `track(initial) => { value, revision, beginChange(), prepareReplace(v), adopt(p) }` | Owns the committed immutable revision of one loaded document | Session cache (`LoadedDocument.tracker`), new incarnations and fork copies in `Transaction` | host |
| `Change<T>` | `{ state: Draft<T>; prepare(): Prepared<T>; abort() }` | One open overlay draft | `Transaction#acquire` | host |
| `Prepared<T>` | `{ base, value, ops: readonly Op[], baseRevision, abort() }` | Candidate revision plus the exact op batch. Adopted after the storage commit succeeds; aborted on failure or when empty. | `Transaction.settleSuccess`/`adopt`/`discard` | host |
| `Op` | `["r", v] \| ["s", path, v] \| ["d", path] \| ["a", path, str] \| ["t", path, n] \| ["p", path, index, remove, items] \| ["m", path, permutation]` | The delta vocabulary. `r` replaces the root. `s` sets a path. `d` deletes (an array element is spliced out). `a` appends to a string. `t` trims `n` characters from the **front** of a string. `p` splices an array. `m` permutes an array (`new[i] = old[perm[i]]`). | Storage deltas (`DocumentContent { kind: "delta", ops }`); the `checkpointWhen(value, ops, info)` predicate (base or delta); `CommitPublication` document changes; view mounts (ops re-prefixed under `["docs", kind]`); `events.ts` reads `a` on `…/generation/message/content/i/text` as `text_delta`, `p` on `content` as `*_start`, and `t`/`a` on a tool slot's `output` as a front trim plus an append; `RETIREMENT_OPERATIONS = [["r", null]]` | host (storage, publication, observation) |
| `Path` / `NonEmptyPath` / `Seg` | `readonly (string \| number)[]` (non-empty for `s/d/a/t`) | Op addressing | `view.ts` `prefixed()`, `events.ts` `startsWith()` | host |
| `applyImmutable` | `(target, ops) => T` | Replay one batch copy-on-write, without mutating `target` | view mount, task graph mount | host |
| `applyImmutableBatches` | `(target, batches) => T` | Replay many batches; only the final result is exposed | memory storage materialization | host |
| `apply` | `(target, ops) => T` | Mutable replay of a batch the caller owns exclusively | SQLite storage materialization from persisted deltas | host |
| `overlap` | `(a, b, scan, probe = 64, maxCandidates = 8) => number` | Length of the longest suffix of `a` that is a prefix of `b`, probing a bounded number of candidates | tool progress estimates how many bytes a commit writes (for `PROGRESS_BYTES_PER_SECOND` pacing) | host (progress pacing) |
| durable `assignJson` (own, `harness/json.ts`) | `(target, key, value)` | Writes leaf by leaf so the tracker emits small `a` ops instead of whole-value sets | partial commits, tool details | host performance detail |

### B.5 Replicated state and watch facilities

| unit | signature | role | where | harness logic or host |
|---|---|---|---|---|
| `replicatedState(source, options?)` | `=> AttachedReplicatedState<T>` | Attaches a read-only, synchronously hydrated state to a source | `Session.documentState()`, `Conversation.viewState()`, `Harness.taskGraph()` | observation (host) |
| `ReplicatedState<T>` | `{ value: T \| undefined; subscribe(listener(value, context, delivery: { kind: "hydrate"\|"update"; sequence })) => unsubscribe }` | Serialized callbacks. At most 100 deliveries wait; on overflow only the newest is kept. A failure is reported in isolation. | frontend subscribes to the conversation view and the task graph | observation |
| `AttachedReplicatedState<T>` | `ReplicatedState<T> & { value: T; dispose() }` | Disposal releases the attachment; the last value stays readable | public `DocumentState<T> = AttachedReplicatedState<Readonly<T> \| null>`, `viewState`, `taskGraph`; frontend `runtime.ts` | observation |
| `ReplicatedStateSource<T>` | `{ attach(): ReplicatedStateSourceAttachment<T> }` | Must capture a snapshot and register buffering atomically, with no gap or overlap | implemented by durable `CommittedStateSource` | host |
| `ReplicatedStateSourceAttachment<T>` | `{ snapshot: { value, cursor }; activate(listener(frame)); dispose() }` | Single-use activation drains buffered frames in commit order | `SessionSourceAttachment` | host |
| `ReplicatedStateSourceFrame<T>` | `{ cursor; value; ops: readonly Op[]; context }` | One committed revision with its exact ops; `cursor = previous + 1` | `CommittedStateSource.advance` | host |
| durable `WatchHandle<T>` / `CommittedWatch` | `{ value; start(listener(value, ops, context)); stop(): Promise<WatchEnd>; closed }` | Serialized exact-frame watch (spec §9.2). At most 100 pending frames; frame 101 replaces the undelivered suffix with `[["r", newest]]`. Ends: `stopped`, `cancelled`, `session_closed`, `retired`, `listener_error`. The acquisition context governs its lifetime. | `watchDoc` (exposed to tasks, hooks and tools through `DocumentObserver`), `Conversation.watch()`, `watchTaskGraph()`, `watchEvents` | observation; **the built-in harness tasks never call `watchDoc`**. Only extensions might. |
| durable `ConversationView` mount | `{ conversation; entries; docs: { "pi.agent", "pi.live", "pi.inbox", "pi.provider", "pi.usage" } }` | One mount per conversation, advanced from commit publications. Document ops are re-prefixed; appended entries become splices. | `viewState`, `watch`, `watchEvents` | observation |
| durable `TaskGraph` mount | `{ tasks: Record<id, TaskGraphNode> }` | Live tasks, their states, owners and owned conversations | `taskGraph`, `watchTaskGraph`, frontend task panel | observation |
| durable `watchEvents` → `AgentEvent` | snapshot + per-commit event batches | Turns before/after views and their ops into coding-agent-style events (`message_update` with text/thinking/toolcall deltas, `tool_execution_update` with output trim/append, …) | experimental UI adapter | observation (UI) |

### B.6 Roles: host or harness, and the natlang replacement

| Chord role | host infrastructure or harness logic | what the harness logic actually needs | natlang mechanism | replaceable? |
|---|---|---|---|---|
| `Context.abortSignal` as a parameter | both | "if the invocation is signalled, stop without writing; the abort phase settles" | task `AbortSignal` (`runtime.run(fn, { signal })`, call-tree cancellation). Abort phases run as separate calls. | yes |
| `withAbortSignal` per invocation | host (scheduler) | — | one natlang task per phase invocation, with its own signal | yes |
| `withoutAbortSignal`, `awaitWithContext`, `BACKGROUND_CONTEXT` | host | — | host code around services (commit settlement and cleanup are not cancellable) | n/a (host keeps them) |
| `JsonValue`, `JsonRepresentation`, `copyJson` | types (both) | strict JSON | typed JSON values; copying is implicit | yes |
| `Draft` via `tx.doc()` | **harness logic** | Read-modify-write of several documents and table rows **in one atomic commit on one serial line** (the Session line) | A commit step is a reducer over a copy of the state. **`EventLoop`** (serial application, duplicate suppression, commit before publish) is the Session line. **Folder transactions** (`folder.apply`) are the commit callback: changes are kept only on success. Documents become typed JSON values (or JSON files in a session folder) that the step returns or edits. | yes; owner decides the layout (B.7) |
| `track` / `Change` / `Prepared` / `adopt` | host | — | the EventLoop commit, or `folder.apply` installing changes on success | yes |
| `Op` / `Path`, `checkpointWhen(value, ops, info)` | host | — | The storage service decides base or delta. Natlang publishes whole states; diffs, if needed, are computed by the host. | yes (no NL exposure) |
| `applyImmutable*`, `apply` | host (storage replay, mounts) | — | none needed when states are stored and published whole; otherwise inside the storage service | yes |
| `overlap`, `assignJson` | host (performance) | — | — | out |
| `replicatedState` / `AttachedReplicatedState` / `ReplicatedState.subscribe` | observation | — | `EventLoop` `view` + `onCommit` (publication after commit). `KeyedEventLoop` cannot carry the Session line because commits cross conversations; per-conversation views can be derived from the single loop's state. | yes (frontend) |
| `ReplicatedStateSource*` attach/activate/frame protocol | host | — | EventLoop publication | yes |
| `WatchHandle`, `watchDoc` (bounded frames, overflow replacement) | observation, extension-facing | not used by built-in tasks | a host subscription service exposed to extensions only if one needs it | partly; exact op frames are lost |
| `watchEvents` / `AgentEvent` from op shapes | observation (UI) | — | Derive events from before/after states. `ai.turn` can publish partial deltas itself. | out or host |
| `ConversationView`, `TaskGraph` mounts | observation | — | views over the EventLoop state | host |

### B.7 Notes and owner decisions

1. **Where the Chord-tracked documents live.** pi-durable's Session is **one** serial mutation line for the whole
   session. A single commit routinely touches several conversations: creating a child conversation with its owner
   task, settling submissions, editing the parent's `pi.live`, writing `pi.usage`. The faithful mapping:
   - one (unkeyed) `EventLoop` is the Session line;
   - each `runtime.commit(change)` is one event whose step runs the change on a copy of the state and commits only on
     success (folder-transaction semantics);
   - the documents are typed JSON values in that state (or files in a session folder applied with `folder.apply`).

   `KeyedEventLoop` per conversation would break cross-conversation atomicity. **Decision:** one EventLoop state vs a
   session folder of JSON documents. The tables (entries, tasks, submissions) need the same choice and the storage
   service's journal, because EventLoop exactly-once needs a persistent journal.
2. **High-frequency progress commits.** Partials are committed every 100 ms and tool output every 100 ms (paced by
   bytes). These must stay host-side writes, i.e. crisp events dispatched to the loop, not natural-language reducer
   steps. **Decision:** `ai.turn` (and the tool-execution service) may write `pi.live` progress directly through the
   host. The other option is to drop live partials from the port.
3. **Which model `ai` drives.** A faithful port puts pi-ai behind `ai` (provider HTTP, the user's model choice per
   conversation, deferred responses). That is different from natlang's own interpreter loop, which the existing
   `applications/pi` used as the agent loop. **Decision:** the agent model runs as an `ai.turn` provider call that the
   natural-language harness orchestrates (pi-durable semantics), or natlang's own driver is the agent. If both,
   `ai.turn` needs a natlang-driver adapter.
4. **Pure helpers.** The overflow and retry tables, token estimates, argument validation and the transcript tool
   replay are pi-ai's definitions. Keeping them as `ai` members keeps one source of truth. Copying them into crisp
   helpers would fork it. Recommended: `ai` for `failure`, `estimateTokens` and `validateArguments`; crisp for the
   three transcript helpers if the generation and prompt functions share a folder.
5. **Exact observation.** The bounded-frame watches with overflow replacement, exact op batches and the
   op-shape-derived UI deltas have no natlang counterpart, and the harness does not need them. Recommended: **out**,
   except a whole-state subscription for the frontend.
