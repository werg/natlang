#!/usr/bin/env node
/**
 * Arm A of the flagship benchmark (plans/semantic-fp/06 §4): a frontier model in a standard tool loop with a prose task
 * prompt. It reads one JSON job on stdin and writes one JSON line per unit on stdout.
 *
 *   job: { provider, model, piOptions?, system, tools: [OpenAI function tools], submit: TOOL_NAME, max_turns,
 *          concurrency, units: [{ id, user }] }
 *
 * Each unit is its own conversation: system + the unit's user message, then model turns. A call of `submit` ends the
 * unit with its arguments as the output; a call of any other offered tool is answered by `job.services` (none for apps
 * without services) and the loop continues; a turn with no tool call is answered with one sentence asking for the
 * submit call. Transport is natlang's Pi provider backend (ts-host/src/model/pi-provider.ts) with the credentials of
 * `natlang auth`; nothing here prints or stores them. Usage is Pi's: uncached input, cache reads and writes, output.
 */
import { createPiModelBackend } from '../../ts-host/dist/model/pi-provider.js';

const read = async stream => { let text = ''; for await (const chunk of stream) text += chunk; return text; };
const job = JSON.parse(await read(process.stdin));
const backend = createPiModelBackend(job.provider, job.model, process.env, undefined, undefined, job.piOptions ?? {});
await backend.prepare();

async function runUnit(unit) {
  const messages = [{ role: 'system', content: job.system }, { role: 'user', content: unit.user }];
  const usage = { input: 0, cache_read: 0, cache_write: 0, output: 0, requests: 0 };
  const started = Date.now();
  const turns = [];
  try {
    for (let turn = 1; turn <= job.max_turns; turn++) {
      const reply = await backend.turn({ messages, tools: job.tools, seed: null, max_tokens: null });
      const raw = reply.raw_response?.usage ?? {};
      usage.requests++;
      usage.input += raw.input ?? 0; usage.cache_read += raw.cacheRead ?? 0;
      usage.cache_write += raw.cacheWrite ?? 0; usage.output += raw.output ?? reply.completion_tokens ?? 0;
      turns.push({ text: reply.text ?? '', calls: reply.calls ?? [], reasoning_chars: (reply.reasoning ?? '').length });
      messages.push({ role: 'assistant', content: reply.text ?? '', tool_calls: reply.raw_calls ?? [] });
      const calls = (reply.raw_calls ?? []);
      if (!calls.length) {
        messages.push({ role: 'user', content: `Submit your answer by calling ${job.submit}.` });
        continue;
      }
      for (const call of calls) {
        const name = call.function.name;
        let args;
        try { args = JSON.parse(call.function.arguments || '{}'); }
        catch { messages.push({ role: 'tool', tool_call_id: call.id, content: 'The arguments were not valid JSON.' }); continue; }
        if (name === job.submit)
          return { id: unit.id, output: args, turns: turn, usage, latency_ms: Date.now() - started, transcript: turns };
        messages.push({ role: 'tool', tool_call_id: call.id, content: `There is no tool ${name}.` });
      }
    }
    return { id: unit.id, output: null, error: 'no-submit', turns: job.max_turns, usage, latency_ms: Date.now() - started, transcript: turns };
  } catch (error) {
    return { id: unit.id, output: null, error: `transport: ${String(error?.message ?? error).slice(0, 300)}`, turns: turns.length,
      usage, latency_ms: Date.now() - started, transcript: turns };
  }
}

const queue = [...job.units];
await Promise.all(Array.from({ length: Math.max(1, job.concurrency ?? 1) }, async () => {
  for (let unit = queue.shift(); unit; unit = queue.shift()) process.stdout.write(JSON.stringify(await runUnit(unit)) + '\n');
}));
backend.close?.();
