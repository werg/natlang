#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, rename, writeFile } from 'node:fs/promises';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const digest = value => createHash('sha256').update(value).digest('hex');

export async function renderSftTurn(turn, render, endToken = '<|im_end|>') {
  if (turn.training_admission?.approved === false) return null;
  if (!Array.isArray(turn.messages) || !Array.isArray(turn.tools) || !turn.target)
    throw new Error(`${turn.id}: missing messages/tools/target training view`);
  const messages = structuredClone(turn.messages), target = structuredClone(turn.target);
  // Whitespace around the reasoning carries nothing, and templates differ in whether they keep it (Ling's strips it).
  const reasoning = turn.teacher_reasoning?.trim();
  if (reasoning) target.reasoning_content = reasoning;
  if (Array.isArray(target.tool_calls)) target.tool_calls = target.tool_calls.map((call, index) =>
    ({ ...call, id: `teacher_${index}` }));
  const prompt = await render(messages, turn.tools);
  const after = target.tool_calls?.length ? target.tool_calls.map(call =>
    ({ role: 'tool', tool_call_id: call.id, content: 'X' })) : [{ role: 'user', content: 'X' }];
  const complete = await render([...messages, target, ...after], turn.tools);
  if (!complete.startsWith(prompt)) throw new Error(`${turn.id}: template changed the assistant prefix`);
  const suffix = complete.slice(prompt.length), end = suffix.indexOf(endToken);
  if (end < 0) throw new Error(`${turn.id}: assistant end token is absent from rendered target`);
  const completion = suffix.slice(0, end + endToken.length);
  if (reasoning && !completion.includes(reasoning))
    throw new Error(`${turn.id}: template dropped teacher reasoning`);
  return { id: turn.id, program_id: turn.program_id, source_groups: turn.source_groups ?? [],
    family: turn.family, skill: turn.skill, renderer: 'server-chat-template',
    context_items: messages.length, teacher_trajectory_id: turn.teacher_trajectory_id,
    teacher_trajectory_digest: turn.teacher_trajectory_digest,
    training_admission: turn.training_admission, prompt, completion };
}

async function main() {
  const args = process.argv.slice(2);
  const positional = args.filter((value, index) => !value.startsWith('--') && !args[index - 1]?.startsWith('--'));
  if (positional.length !== 2)
    throw new Error('usage: export-native-sft.mjs INPUT.jsonl OUTPUT.jsonl [--server URL] [--workers N] [--end-token TOKEN]');
  const take = (flag, fallback) => { const index = args.indexOf(flag); return index < 0 ? fallback : args[index + 1]; };
  const input = resolve(positional[0]), output = resolve(positional[1]);
  const server = take('--server', 'http://127.0.0.1:8081'), workers = Number(take('--workers', '4'));
  const endToken = take('--end-token', '<|im_end|>');
  if (!Number.isInteger(workers) || workers < 1) throw new Error('workers must be positive');
  const propsResponse = await fetch(`${server}/props`);
  const props = propsResponse.ok ? await propsResponse.json() : {};
  const template = props.chat_template_tool_use ?? props.chat_template ?? null;
  const render = async (messages, tools) => {
    // The server closes idle keep-alive connections, and a request can race that close. Rendering is idempotent, so a
    // request that failed before any response is sent again; HTTP errors are not retried.
    const post = () => fetch(`${server}/apply-template`, { method: 'POST',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages, tools }) });
    let response;
    for (let attempt = 1; ; attempt++) {
      try { response = await post(); break; }
      catch (error) { if (attempt >= 3 || !(error instanceof TypeError)) throw error; }
    }
    if (!response.ok) throw new Error(`template server returned HTTP ${response.status}`);
    const body = await response.json();
    if (typeof body.prompt !== 'string') throw new Error('template server returned no prompt');
    return body.prompt;
  };
  // Streamed: a corpus's turns (each with its whole context) outgrow a single string. Up to `workers` render at
  // once, and rows are written in input order.
  await mkdir(dirname(output), { recursive: true });
  const staged = `${output}.building-${process.pid}-${randomUUID()}`;
  const sink = createWriteStream(staged, { flags: 'wx' });
  const write = async text => { if (!sink.write(text)) await once(sink, 'drain'); };
  let rows = 0, turns = 0;
  const inFlight = [];
  const settle = async () => {
    const row = await inFlight.shift();
    if (row) { rows++; await write(JSON.stringify(row) + '\n'); }
  };
  for await (const line of createInterface({ input: createReadStream(input), crlfDelay: Infinity })) {
    if (!line) continue;
    const turn = JSON.parse(line);
    turns++;
    inFlight.push(renderSftTurn(turn, render, endToken).catch(error => {
      throw new Error(`${turn.id}: ${error.message}${error.cause ? ` (${error.cause.code ?? ''} ${error.cause.message ?? error.cause})` : ''}`, { cause: error });
    }));
    if (inFlight.length >= workers) await settle();
  }
  while (inFlight.length) await settle();
  sink.end(); await once(sink, 'finish');
  await rename(staged, output);
  const sourceHash = createHash('sha256');
  for await (const chunk of createReadStream(input)) sourceHash.update(chunk);
  await writeFile(`${output}.manifest.json`, JSON.stringify({ version: 'natlang.sft.native/1',
    source: input, source_sha256: sourceHash.digest('hex'), rows,
    skipped_unapproved: turns - rows, renderer: { server,
      template_sha256: typeof template === 'string' ? digest(template) : null, end_token: endToken } }, null, 2) + '\n');
  console.log(`${rows} approved SFT turns -> ${output}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; });
