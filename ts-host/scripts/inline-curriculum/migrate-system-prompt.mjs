#!/usr/bin/env node
/**
 * Give recorded contexts the system prompt a call is served with now, whichever earlier version they were collected
 * with.
 *
 *   node scripts/inline-curriculum/migrate-system-prompt.mjs IN.results.jsonl OUT.results.jsonl
 *
 * A call's system prompt is TOOLS_PROMPT, then APPROACH_PROMPT when the collector ran with --approach-guide, then
 * FUNCTION_TOOLS_PROMPT when the call has functions of its own, then DIRECTORY_REDUCER_PROMPT for a folder call
 * (native/agent.ts). Every version of those parts in the history of native/prompt.ts is read from git (also as
 * migrate-code-tools.mjs renamed the code tools in it); a system
 * message made of versions of them is rebuilt from the current parts, keeping the function-tools and folder parts it
 * had and leaving out the approach guide, which the student is not served with. A row with a system message not made of
 * known parts was collected under a prompt, and a runtime, that is not recorded: it is left out and counted. Any row can
 * be passed again; each rewritten row is marked in provenance.system_prompt_migration.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { DIRECTORY_REDUCER_PROMPT, FUNCTION_TOOLS_PROMPT, TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { migrateIrLine as renameCodeTools } from './migrate-code-tools.mjs';

const MIGRATION = 'current-prompt/1';
const REPO = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const FILE = 'ts-host/src/native/prompt.ts';

/** Every version of each prompt part in the history of native/prompt.ts, longest first. */
export async function promptHistory() {
  const parts = { TOOLS_PROMPT: new Set(), APPROACH_PROMPT: new Set(), FUNCTION_TOOLS_PROMPT: new Set(),
    DIRECTORY_REDUCER_PROMPT: new Set() };
  const revisions = execFileSync('git', ['log', '--format=%H', '--', FILE], { cwd: REPO, encoding: 'utf8' }).split('\n').filter(Boolean);
  for (const revision of revisions) {
    const source = execFileSync('git', ['show', `${revision}:${FILE}`], { cwd: REPO, encoding: 'utf8', maxBuffer: 1 << 26 });
    const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    const module = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
    // A row migrated to the read_code tool names (migrate-code-tools.mjs) has each version in that form too.
    for (const [name, versions] of Object.entries(parts)) if (typeof module[name] === 'string')
      versions.add(module[name]).add(JSON.parse(renameCodeTools(JSON.stringify(module[name]))));
  }
  return Object.fromEntries(Object.entries(parts).map(([name, versions]) =>
    [name, [...versions].filter(Boolean).sort((a, b) => b.length - a.length)]));
}

/** The current prompt for a system message made of known parts, or undefined when it is not. */
export function currentPrompt(message, history) {
  const take = (text, versions) => {
    const found = versions.find(version => text.startsWith(version));
    return found === undefined ? [false, text] : [true, text.slice(found.length)];
  };
  for (const base of history.TOOLS_PROMPT) {
    if (!message.startsWith(base)) continue;
    let rest = message.slice(base.length), functions, folder;
    [, rest] = take(rest, history.APPROACH_PROMPT);
    [functions, rest] = take(rest, history.FUNCTION_TOOLS_PROMPT);
    [folder, rest] = take(rest, history.DIRECTORY_REDUCER_PROMPT);
    if (rest === '') return TOOLS_PROMPT + (functions ? FUNCTION_TOOLS_PROMPT : '') + (folder ? DIRECTORY_REDUCER_PROMPT : '');
  }
  return undefined;
}

/** The row with current system prompts, or null when one of its system messages is not made of known parts. */
export function migrateRow(row, history) {
  let changed = false, unknown = false;
  const trajectory = (row.trajectory ?? []).map(turn => {
    if (!Array.isArray(turn.context)) return turn;
    let turnChanged = false;
    const context = turn.context.map(message => {
      if (message.role !== 'system' || typeof message.content !== 'string') return message;
      const current = currentPrompt(message.content, history);
      if (current === undefined) { unknown = true; return message; }
      if (current === message.content) return message;
      turnChanged = true;
      return { ...message, content: current };
    });
    if (!turnChanged) return turn;
    changed = true;
    return { ...turn, context };
  });
  if (unknown) return null;
  if (!changed) return row;
  return { ...row, trajectory, provenance: { ...row.provenance, system_prompt_migration: MIGRATION } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: migrate-system-prompt.mjs IN.results.jsonl OUT.results.jsonl');
  const history = await promptHistory();
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  let changed = 0, dropped = 0;
  const out = lines.flatMap(line => {
    const row = JSON.parse(line), next = migrateRow(row, history);
    if (next === null) { dropped++; return []; }
    if (next === row) return [line];
    changed++;
    return [JSON.stringify(next)];
  });
  writeFileSync(output, out.join('\n') + '\n');
  process.stdout.write(`${changed}/${lines.length} rows given the current system prompt; ${dropped} with a prompt not ` +
    `made of known parts left out -> ${output}\n`);
}
