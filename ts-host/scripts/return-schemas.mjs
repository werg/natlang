#!/usr/bin/env node
// The return_result `value` schema ts-host offers for a call (NativeToolAgent.toolsScope: schemaOf(lam.type.returns)),
// computed from the call's opening user message ("You are inside this call: <signature>" and the `type` aliases listed
// under it). Used by the training renderer (training/neuralese/natlang_neuralese/data/serving_prompt.py) for recorded
// calls whose tool schemas were not recorded.
//
//   node scripts/return-schemas.mjs [DIST_DIR]      (stdin: JSON lines {"user": ...}; stdout: {"schema"} or {"error"})
import { createInterface } from 'node:readline';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dist = resolve(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), '../dist'));
const { parseType, TypeEnv } = await import(pathToFileURL(join(dist, 'native/types.js')).href);
const { schemaOf } = await import(pathToFileURL(join(dist, 'native/agent.js')).href);

const CLOSE = { '(': ')', '[': ']', '{': '}', '<': '>' };

/** The declared result type text of `name(params): Result`. */
function returnText(signature) {
  const open = signature.indexOf('(');
  if (open < 0) throw new Error('signature has no parameter list');
  const stack = [];
  for (let i = open; i < signature.length; i++) {
    const ch = signature[i];
    if (ch === '"' || ch === "'") {  // string literal types
      const end = signature.indexOf(ch, i + 1);
      if (end < 0) throw new Error('unterminated string literal in signature');
      i = end;
    } else if (CLOSE[ch]) stack.push(CLOSE[ch]);
    else if (ch === stack.at(-1) && !(ch === '>' && signature[i - 1] === '=')) {
      stack.pop();
      if (!stack.length) {
        const rest = signature.slice(i + 1);
        if (!rest.startsWith(': ')) throw new Error('signature has no declared result');
        return rest.slice(2);
      }
    }
  }
  throw new Error('unbalanced signature');
}

function schemaFor(user) {
  const lines = user.split('\n');
  const head = 'You are inside this call: ';
  if (!lines[0]?.startsWith(head)) throw new Error('not a call opening');
  const names = {};
  for (const line of lines.slice(1)) {
    if (!line.trim()) break;
    const alias = /^type ([A-Za-z_$][\w$]*) = ([\s\S]*);$/.exec(line);
    if (alias) names[alias[1]] = parseType(alias[2]);
  }
  return schemaOf(parseType(returnText(lines[0].slice(head.length))), new TypeEnv(names));
}

for await (const line of createInterface({ input: process.stdin })) {
  let out;
  try { out = { schema: schemaFor(JSON.parse(line).user) }; } catch (error) { out = { error: String(error?.message ?? error) }; }
  process.stdout.write(JSON.stringify(out) + '\n');
}
