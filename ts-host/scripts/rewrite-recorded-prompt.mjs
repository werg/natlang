#!/usr/bin/env node
// What this checkout sends for a recorded tool-loop call, for the prompt cache map's re-run on the same records
// (scripts/neuralese_data/prompt_cache_map.py --rewrite): stdin JSONL {messages, tools} -> stdout JSONL {messages, tools}.
// The system message becomes this runtime's SYSTEM_PROMPT, the tool list the one NativeToolAgent offers the same call
// surface (servingTools: own code, folder access and file-tool surface read off the recorded tools, depth limit off the
// opening, return_result typed by the opening's signature or, failing that, by the recording's own value schema), and
// the 2026-11-22 texts of the folder and function-tools notes in the opening their current texts. Other calls (decision
// readouts, predicate judges) pass through unchanged.
import { createInterface } from 'node:readline';
import { SYSTEM_PROMPT, FUNCTION_TOOLS_PROMPT, NL_DEPTH_LIMIT_NOTICE, directoryReducerPrompt } from '../dist/native/prompt.js';
import * as previous from '../dist/native/prompt-2026-11-22.js';
import { declaredResult, servingTools } from '../dist/teacher/serving-request.js';

const INTERPRETER = 'You are running one call of a natural-language function. Execute the function to completion';
const notes = [];
for (const surface of ['all', 'editor', 'files']) for (const adHoc of [true, false]) for (const writable of [true, false])
  notes.push([previous.directoryReducerPrompt(surface, adHoc, writable), directoryReducerPrompt(surface, adHoc, writable)]);
notes.push([previous.FUNCTION_TOOLS_PROMPT, FUNCTION_TOOLS_PROMPT]);
notes.sort((a, b) => b[0].length - a[0].length);

function rewrite({ messages, tools = [] }) {
  const system = messages[0];
  if (system?.role !== 'system' || typeof system.content !== 'string' || !system.content.includes(INTERPRETER)) return { messages, tools };
  const names = new Set(tools.map(tool => (tool.function ?? tool).name));
  const writes = ['write_file', 'edit_file', 'editor', 'bash', 'python'].some(name => names.has(name));
  const folder = names.has('list_files') || names.has('read_file') || names.has('editor') || writes ? (writes ? 'write' : 'read') : null;
  const fileTools = names.has('editor') && !names.has('list_files') ? 'editor' : names.has('list_files') && !names.has('bash') ? 'files' : 'all';
  const openingAt = messages.findIndex(message => message.role === 'user');
  let opening = typeof messages[openingAt]?.content === 'string' ? messages[openingAt].content : '';
  const surface = { ownCode: names.has('edit_code'), folder, fileTools, depthLimited: opening.includes(NL_DEPTH_LIMIT_NOTICE) };
  let offered;
  try { offered = servingTools(surface, declaredResult(opening)); } catch {
    offered = servingTools(surface);
    const recorded = tools.map(tool => tool.function ?? tool).find(fn => fn.name === 'return_result')?.parameters?.properties?.value;
    if (recorded) {
      const last = offered.at(-1).function.parameters.properties;
      const { description: _description, ...typed } = recorded;
      last.value = { ...typed, description: last.value.description };
    }
  }
  for (const [old, now] of notes) if (old && old !== now) opening = opening.split(old.trim()).join(now.trim());
  const out = messages.map((message, index) => index === 0 ? { ...message, content: SYSTEM_PROMPT } :
    index === openingAt ? { ...message, content: opening } : message);
  return { messages: out, tools: offered };
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) if (line.trim()) process.stdout.write(JSON.stringify(rewrite(JSON.parse(line))) + '\n');
