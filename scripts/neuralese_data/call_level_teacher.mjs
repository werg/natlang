#!/usr/bin/env node
// Engine A's teacher transport and type check (plans/NATLANG_COMPILER.md §8.1; driven by call_level_engine.py teach).
//
// One JSON request per stdin line, one JSON answer per stdout line (any order; `id` pairs them):
//   {"op": "chat", "id", "endpoint", "chat_completions_url", "model", "api_key", "chat_config", "messages", "tools",
//    "temperature", "max_tokens", "check": {"returns": TYPE_TEXT, "declarations": TS_TEXT} | null}
//     -> {"id", "turn": {calls, text, reasoning, truncated, malformed_call, prompt_tokens, completion_tokens},
//         "check": {"checked": bool, "ok": bool, "problem"?, "value"?}} | {"id", "error"}
//   {"op": "check", "id", "value", "check": {...}} -> {"id", "check": {...}}
// The request goes through the runtime's own chat client (ts-host model/chat-completion.ts: Gemini's host-call
// signatures and thought summaries, reasoning history aliases, malformed-call retry), so a teacher sees exactly the
// request the collector would send. The type check is the runtime's own (`coerce` against the call's declared return
// type with the declarations rendered in its opening message), the same check a returned value meets at run time.
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

const runtime = process.env.NATLANG_TS_HOST || join(new URL('.', import.meta.url).pathname, '../../ts-host');
const load = p => import(pathToFileURL(join(runtime, 'dist', p)).href);
const [chat, types, values, tsModule] = await Promise.all([load('model/chat-completion.js'), load('native/types.js'),
  load('native/values.js'), import(pathToFileURL(join(runtime, 'node_modules/typescript/lib/typescript.js')).href)]);
const ts = tsModule.default ?? tsModule;
const concurrency = Math.max(1, Number(process.env.CALL_LEVEL_CONCURRENCY || 2));
const minIntervalMs = Math.max(0, Number(process.env.CALL_LEVEL_MIN_INTERVAL_MS || 0));

/** `type Name = ...;` aliases of a rendered call's declarations, as type text by name (TypeScript's own parser). */
export function declaredAliases(text) {
  const source = ts.createSourceFile('call.ts', text || '', ts.ScriptTarget.Latest, true);
  const out = {};
  for (const statement of source.statements)
    if (ts.isTypeAliasDeclaration(statement)) out[statement.name.text] = statement.type.getText(source);
    else if (ts.isInterfaceDeclaration(statement))
      out[statement.name.text] = '{ ' + statement.members.map(m => m.getText(source).replace(/[;,]\s*$/, '')).join('; ') + ' }';
  return out;
}

/** The runtime's check of a value against the call's declared return type: `checked` false when the type or its
 * declarations do not parse (no checker verdict, never a pass). `Neuralese<T>` is checked as T (the value a writer
 * materializes). */
export function checkReturn(value, check) {
  if (!check || !check.returns) return { checked: false, reason: 'no declared return type' };
  let text = check.returns.trim();
  const neuralese = /^Neuralese<([\s\S]*)>$/.exec(text);
  if (neuralese) text = neuralese[1].trim();
  let type, env;
  try {
    type = types.parseType(text);
    env = new types.TypeEnv(types.parseAliases(declaredAliases(check.declarations)));
  } catch (error) { return { checked: false, reason: `type does not parse: ${String(error.message || error).slice(0, 200)}` }; }
  try { values.coerce(value, type, env, 'value'); }
  catch (error) {
    if (error instanceof values.Reject) return { checked: true, ok: false, problem: error.message.slice(0, 500) };
    return { checked: false, reason: `checker error: ${String(error.message || error).slice(0, 200)}` };
  }
  return { checked: true, ok: true };
}

let last = 0;
async function paced() {
  const wait = last + minIntervalMs - Date.now();
  last = Math.max(Date.now(), last + minIntervalMs);
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
}

async function answer(request) {
  if (request.op === 'check') return { id: request.id, check: checkReturn(request.value, request.check) };
  await paced();
  const transport = chat.httpChatTransport({ endpoint: request.endpoint, model: request.model,
    ...(request.api_key ? { apiKey: request.api_key } : {}),
    ...(request.chat_completions_url ? { chatCompletionsUrl: request.chat_completions_url } : {}) });
  const config = { ...(request.chat_config || {}), ...(request.max_tokens ? { max_tokens: request.max_tokens } : {}) };
  const turn = await chat.chatCompletionModelTurn(transport, { request: config })(
    { messages: request.messages, tools: request.tools || [], temperature: request.temperature, seed: request.seed ?? null });
  const out = { calls: turn.calls || [], text: turn.text || '', reasoning: turn.reasoning || null,
    truncated: !!turn.truncated, malformed_call: turn.malformed_call || null,
    prompt_tokens: turn.prompt_tokens ?? null, completion_tokens: turn.completion_tokens ?? null };
  return { id: request.id, turn: out };
}

const main = import.meta.url === pathToFileURL(process.argv[1]).href;
if (main) {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const running = new Set();
  for await (const line of lines) {
    if (!line.trim()) continue;
    const request = JSON.parse(line);
    while (running.size >= concurrency) await Promise.race(running);
    const job = answer(request).catch(error => ({ id: request.id, error: String(error?.message || error).slice(0, 2000),
      ...(error?.status ? { status: error.status } : {}) }))
      .then(result => { process.stdout.write(JSON.stringify(result) + '\n'); });
    running.add(job);
    job.finally(() => running.delete(job));
  }
  await Promise.all(running);
}
