#!/usr/bin/env node
// Synthesized reasoning for static demonstrations (owner 2026-10-05): every scripted turn of a reference replay (root
// steps and the answers of nl children) gets a short first-person rationale written by the teacher model from what the
// turn can see and the action it takes, instead of a stock action note.
// Owner 2026-10-06: only root turns (planning, delegation, answer synthesis) are rationalized. Turns inside nl children
// (judging or extracting from the item they were handed) act without reasoning: empty, trained as acting directly, no
// teacher request. The answer is the judgment; a sentence restating it adds nothing the demonstration lacks.
//   node rationales.mjs apply-policy PROMPTS.jsonl OLD.rationales.jsonl NEW.rationales.jsonl   (earlier collections)
//   node rationales.mjs collect CASES.ir.jsonl PROMPTS.jsonl          replay once, record each turn's view and action
//   node rationales.mjs generate PROMPTS.jsonl RATIONALES.jsonl [--server URL] [--model ID] [--concurrency N]
// then replay-demonstrations.mjs CASES ROWS TURNS --rationales RATIONALES.jsonl. Keys do not depend on earlier
// reasoning (it is part of later contexts): case id, the user and tool messages, the turn's index in its call, the action.
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import * as collector from '../../dist/teacher/collector.js';
import * as curriculum from '../../dist/teacher/curriculum.js';

const text = content => typeof content === 'string' ? content : Array.isArray(content) ? content.map(part => part.text ?? '').join('') : '';
const clip = (value, n) => value.length > n ? value.slice(0, n) + ' …' : value;
// Keeps both ends: the runtime's opening eval declares arguments and captured variables after a long interface preamble.
const clipMiddle = (value, n) => value.length > n ? value.slice(0, n / 4) + ' … ' + value.slice(-(n * 3 / 4)) : value;

/** The stable identity of a scripted turn: everything it was shown except earlier reasoning. The tool results count:
 * sibling children of one lambda share their opening and differ only in the arguments their first result declares. */
export function rationaleKey(caseId, context, calls) {
  const seen = context.filter(message => message.role === 'user' || message.role === 'tool').map(message => text(message.content));
  const turn = context.filter(message => message.role === 'assistant').length;
  return createHash('sha256').update(JSON.stringify([caseId, seen, turn, calls])).digest('hex').slice(0, 32);
}

/** What the rationale writer sees: everything the agent sees except the system prompt and earlier reasoning — the
 * call's opening, the runtime's scripted declaration of its arguments and captured variables (their values live in that
 * first eval's code), every action and result, then the action taken. Long items are clipped, the middle first. */
function view(context, calls) {
  const items = context.filter(message => message.role !== 'system').flatMap(message => {
    if (message.role === 'tool') return [`[result] ${clipMiddle(text(message.content), 1600)}`];
    if (message.role === 'assistant') return (message.tool_calls ?? []).map(call => `[action] ${call.function?.name} ${clipMiddle(call.function?.arguments ?? '', 2400)}`);
    return [clip(text(message.content), 3500)];
  });
  const kept = items.length > 9 ? [...items.slice(0, 4), `[… ${items.length - 9} earlier steps omitted …]`, ...items.slice(-5)] : items;
  return { opening: kept.join('\n\n'), results: [], action: calls.map(([tool, args]) => ({ tool, arguments: args })) };
}

/** Whether a turn belongs to an nl child rather than the case's root call. */
const isChild = (ir, context) => curriculum.callName(context) !== ir.semantics.root.replace(/\.nl$/, '').split('/').pop();

export function loadRationales(path) {
  return new Map(readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)).map(r => [r.key, r.text]));
}

async function collect(input, output) {
  const cases = readFileSync(input, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const seen = new Set();
  writeFileSync(output, '');
  for (const ir of cases) {
    await curriculum.replayReference(ir, collector.defaultSystemPrompt, (context, calls) => {
      const key = rationaleKey(ir.id, context, calls);
      // A child's opening read of the one file it was handed has nothing to reason about: no model call, no reasoning.
      // (A child's context opens with scripted assistant/tool messages carrying its inputs, so "first read" is the test.)
      const child = isChild(ir, context);
      const mechanical = child && calls.length === 1 && calls[0][0] === 'read_file' &&
        !context.some(message => message.tool_calls?.some(call => call.function?.name === 'read_file'));
      if (!seen.has(key)) { seen.add(key); appendFileSync(output, JSON.stringify({ key, case: ir.id, ...view(context, calls), ...(mechanical ? { mechanical: true } : {}), ...(child ? { child: true } : {}) }) + '\n'); }
      return undefined;
    });
  }
  console.log(JSON.stringify({ cases: cases.length, turns: seen.size }));
}

const SYSTEM = `You write the brief private reasoning an agent has just before it acts. You see what the agent sees (the call it is inside, with its instructions and inputs, and the latest tool results) and the action it then takes. Write 1-4 sentences in the first person, present tense, that lead naturally to that action: what matters in what it sees, and why this step comes next. When the call opens, sketch the plan of execution the action begins. For a final answer, point to the evidence in the inputs that supports it. When the action is mechanical and nothing sensible can be said for it (reading the one file the call names, returning a value just computed), output exactly NONE. Never mention hidden labels, references, grading, or that the action was given to you; do not restate the code. Output only the reasoning.`;

async function generate(input, output, flags) {
  const server = flags.server ?? 'http://127.0.0.1:8082', model = flags.model ?? 'nvidia/Qwen3.6-35B-A3B-NVFP4';
  const done = new Set(existsSync(output) ? readFileSync(output, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line).key) : []);
  const queue = readFileSync(input, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)).filter(p => !done.has(p.key));
  const silent = p => p.mechanical || p.child;
  for (const p of queue.filter(silent)) appendFileSync(output, JSON.stringify({ key: p.key, case: p.case, text: '' }) + '\n');
  queue.splice(0, queue.length, ...queue.filter(p => !silent(p)));
  let ok = 0, failed = 0;
  async function worker() {
    while (queue.length) {
      const p = queue.shift();
      const user = `The call and its inputs:\n${p.opening}\n\n${p.results.length ? `Latest results:\n${p.results.join('\n---\n')}\n\n` : ''}The action taken next:\n${JSON.stringify(p.action).slice(0, 2500)}`;
      // A busy or restarting server is retried with backoff; a prompt that still fails is left for the next run.
      for (let attempt = 0; attempt < 5; attempt++) {
        try {
          const response = await fetch(`${server}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(600_000),
            body: JSON.stringify({ model, messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }],
              max_tokens: 220, temperature: 0.6, chat_template_kwargs: { enable_thinking: false } }) });
          const body = await response.json();
          const out = body.choices?.[0]?.message?.content?.trim();
          if (!response.ok || !out) throw new Error(`${response.status} ${JSON.stringify(body).slice(0, 200)}`);
          // NONE: the turn acts without reasoning (empty, not a stock note), and stays model-rationalized.
          appendFileSync(output, JSON.stringify({ key: p.key, case: p.case, text: out === 'NONE' ? '' : out }) + '\n');
          ok++;
          break;
        } catch (error) {
          if (attempt === 4) { failed++; if (failed % 50 === 1) console.error(String(error)); }
          else await new Promise(resolve => setTimeout(resolve, 5000 * 2 ** attempt));
        }
      }
    }
  }
  await Promise.all(Array.from({ length: Number(flags.concurrency ?? 48) }, worker));
  console.log(JSON.stringify({ written: ok, failed }));
}

/** Apply the child policy to rationales written before it: child turns become empty, root turns keep their text. */
function applyPolicy(prompts, input, output) {
  const child = new Map(readFileSync(prompts, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line))
    .map(p => [p.key, Boolean(p.child || p.mechanical)]));
  let emptied = 0, kept = 0, unknown = 0;
  writeFileSync(output, '');
  for (const line of readFileSync(input, 'utf8').split('\n').filter(Boolean)) {
    const r = JSON.parse(line);
    if (!child.has(r.key)) unknown++;
    if (child.get(r.key)) { if (r.text) emptied++; r.text = ''; } else if (r.text) kept++;
    appendFileSync(output, JSON.stringify(r) + '\n');
  }
  console.log(JSON.stringify({ emptied, kept_root: kept, unknown }));
}

const main = import.meta.url === pathToFileURL(process.argv[1]).href;
const [mode, a, b, ...rest] = main ? process.argv.slice(2) : [];
const flags = Object.fromEntries(rest.map((v, i) => v.startsWith('--') ? [v.slice(2), rest[i + 1]] : null).filter(Boolean));
if (mode === 'collect') await collect(a, b);
else if (mode === 'generate') await generate(a, b, flags);
else if (mode === 'apply-policy') applyPolicy(a, b, rest[0]);
else if (mode) throw new Error('usage: rationales.mjs collect CASES PROMPTS | generate PROMPTS RATIONALES [--server URL] | apply-policy PROMPTS OLD NEW');
