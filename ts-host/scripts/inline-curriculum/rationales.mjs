#!/usr/bin/env node
// Synthesized reasoning for static demonstrations (owner 2026-10-05): every scripted turn of a reference replay (root
// steps and the answers of nl children) gets a short first-person rationale written by the teacher model from what the
// turn can see and the action it takes, instead of a stock action note.
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

/** The stable identity of a scripted turn: everything it was shown except earlier reasoning. The tool results count:
 * sibling children of one lambda share their opening and differ only in the arguments their first result declares. */
export function rationaleKey(caseId, context, calls) {
  const seen = context.filter(message => message.role === 'user' || message.role === 'tool').map(message => text(message.content));
  const turn = context.filter(message => message.role === 'assistant').length;
  return createHash('sha256').update(JSON.stringify([caseId, seen, turn, calls])).digest('hex').slice(0, 32);
}

/** What the rationale writer sees: the call's opening, the latest results, and the action taken. */
function view(context, calls) {
  const opening = text(context.find(message => message.role === 'user')?.content);
  const results = context.filter(message => message.role === 'tool').slice(-2).map(message => clip(text(message.content), 1500));
  return { opening: clip(opening, 3500), results, action: calls.map(([tool, args]) => ({ tool, arguments: args })) };
}

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
      if (!seen.has(key)) { seen.add(key); appendFileSync(output, JSON.stringify({ key, case: ir.id, ...view(context, calls) }) + '\n'); }
      return undefined;
    });
  }
  console.log(JSON.stringify({ cases: cases.length, turns: seen.size }));
}

const SYSTEM = `You write the brief private reasoning an agent has just before it acts. You see what the agent sees (the call it is inside, with its instructions and inputs, and the latest tool results) and the action it then takes. Write 1-4 sentences in the first person, present tense, that lead naturally to that action: what matters in what it sees, and why this step comes next. For a final answer, point to the evidence in the inputs that supports it. Never mention hidden labels, references, grading, or that the action was given to you; do not restate the code. Output only the reasoning.`;

async function generate(input, output, flags) {
  const server = flags.server ?? 'http://127.0.0.1:8082', model = flags.model ?? 'nvidia/Qwen3.6-35B-A3B-NVFP4';
  const done = new Set(existsSync(output) ? readFileSync(output, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line).key) : []);
  const queue = readFileSync(input, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)).filter(p => !done.has(p.key));
  let ok = 0, failed = 0;
  async function worker() {
    while (queue.length) {
      const p = queue.shift();
      const user = `The call and its inputs:\n${p.opening}\n\n${p.results.length ? `Latest results:\n${p.results.join('\n---\n')}\n\n` : ''}The action taken next:\n${JSON.stringify(p.action).slice(0, 2500)}`;
      try {
        const response = await fetch(`${server}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model, messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }],
            max_tokens: 220, temperature: 0.6, chat_template_kwargs: { enable_thinking: false } }) });
        const body = await response.json();
        const out = body.choices?.[0]?.message?.content?.trim();
        if (!response.ok || !out) throw new Error(`${response.status} ${JSON.stringify(body).slice(0, 200)}`);
        appendFileSync(output, JSON.stringify({ key: p.key, case: p.case, text: out }) + '\n');
        ok++;
      } catch (error) { failed++; if (failed % 50 === 1) console.error(String(error)); }
    }
  }
  await Promise.all(Array.from({ length: Number(flags.concurrency ?? 48) }, worker));
  console.log(JSON.stringify({ written: ok, failed }));
}

const main = import.meta.url === pathToFileURL(process.argv[1]).href;
const [mode, a, b, ...rest] = main ? process.argv.slice(2) : [];
const flags = Object.fromEntries(rest.map((v, i) => v.startsWith('--') ? [v.slice(2), rest[i + 1]] : null).filter(Boolean));
if (mode === 'collect') await collect(a, b);
else if (mode === 'generate') await generate(a, b, flags);
else if (mode) throw new Error('usage: rationales.mjs collect CASES PROMPTS | generate PROMPTS RATIONALES [--server URL]');
