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
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { DIRECTORY_REDUCER_PROMPT, FUNCTION_TOOLS_PROMPT, TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { promptPieces } from '../../dist/native/system-prompts.js';
import { migrateIrLine as renameCodeTools } from './migrate-code-tools.mjs';

const MIGRATION = 'current-prompt/1';
const REPO = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const FILE = 'ts-host/src/native/prompt.ts';
const EMULATION_FILE = 'ts-host/src/model/text-neuralese-emulation.ts';

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

/**
 * Exact historical prompt pieces from native/prompt.ts, with the same scope as the current runtime registry.
 * The source commit identifies the original text; code-tool-renamed variants are derived by the established
 * code-tool migrator and say so explicitly. This is a data export helper, not a second prompt renderer.
 */
export async function historicalPromptPieces() {
  const scopes = new Map(promptPieces().map(({ id, scope }) => [id, scope]));
  const revisions = execFileSync('git', ['log', '--format=%H', '--', FILE], { cwd: REPO, encoding: 'utf8' }).split('\n').filter(Boolean);
  const byText = new Map();
  const add = (commit, logicalId, text) => {
    if (typeof text !== 'string' || !text.trim()) return;
    const scope = scopes.get(logicalId);
    if (!scope) return;
    const variants = [{ text, variant: 'source' }];
    const renamed = JSON.parse(renameCodeTools(JSON.stringify(text)));
    if (renamed !== text) variants.push({ text: renamed, variant: 'code-tools-renamed' });
    for (const item of variants) {
      const sha256 = createHash('sha256').update(item.text, 'utf8').digest('hex');
      const key = `${scope}\0${item.text}`;
      // Revisions are newest first, so identical source text retains the newest exact provenance.
      if (!byText.has(key)) byText.set(key, {
        id: `historical/${logicalId}/${sha256.slice(0, 16)}`, scope, text: item.text,
        commit, historical: true, sha256,
        ...(item.variant === 'source' ? {} : { variant: item.variant }),
      });
    }
  };
  for (const commit of revisions) {
    const source = execFileSync('git', ['show', `${commit}:${FILE}`], { cwd: REPO, encoding: 'utf8', maxBuffer: 1 << 26 });
    const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    const module = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
    add(commit, 'interpreter', module.TOOLS_PROMPT);
    add(commit, 'interpreter/depth-limit', module.TOOLS_PROMPT_AT_NL_DEPTH_LIMIT);
    add(commit, 'depth-limit-notice', module.NL_DEPTH_LIMIT_NOTICE);
    add(commit, 'directory-reducer/declaration', module.DIRECTORY_REDUCER_DECLARATION);
    add(commit, 'neuralese/text-emulation', module.TEXT_NEURALESE_EMULATION_PROMPT);
    if (typeof module.SEMANTIC_RESULT_PROMPT === 'string')
      add(commit, 'neuralese/string-boundaries', module.SEMANTIC_RESULT_PROMPT.split('\n\n')[1]);
    if (typeof module.FUNCTION_TOOLS_PROMPT === 'string') {
      add(commit, 'function-tools/edit-capability', module.FUNCTION_TOOLS_PROMPT.split('\n')[1]);
      add(commit, 'function-tools', module.FUNCTION_TOOLS_PROMPT);
    }
    add(commit, 'generation-guidance', module.GENERATION_GUIDANCE);
    if (module.GENERATION_GUIDANCE && module.TOOLS_PROMPT && module.TOOLS_PROMPT_AT_NL_DEPTH_LIMIT &&
        typeof module.promptAtNlDepthLimit === 'function') {
      const depthPrompt = module.promptAtNlDepthLimit(module.TOOLS_PROMPT + module.GENERATION_GUIDANCE);
      if (depthPrompt.startsWith(module.TOOLS_PROMPT_AT_NL_DEPTH_LIMIT))
        add(commit, 'generation-guidance', depthPrompt.slice(module.TOOLS_PROMPT_AT_NL_DEPTH_LIMIT.length));
    }
    add(commit, 'approach', module.APPROACH_PROMPT);
    add(commit, 'approach', module.APPROACH_PROMPT_AT_NL_DEPTH_LIMIT);
    add(commit, 'directory-reducer/all', module.DIRECTORY_REDUCER_PROMPT);
    add(commit, 'compaction-notice', module.COMPACTION_NOTICE);
    add(commit, 'automatic-note', module.AUTOMATIC_NOTE);
    add(commit, 'handover/open', module.HANDOVER_NOTE_OPEN);
    add(commit, 'handover/close', module.HANDOVER_NOTE_CLOSE);
    add(commit, 'last-turn-notice', module.LAST_TURN_NOTICE);
    add(commit, 'compacted-result', module.COMPACTED_RESULT);
    if (typeof module.directoryReducerPrompt === 'function') {
      const surfaces = module.FILE_TOOL_SURFACES ?? ['all'];
      for (const surface of surfaces) {
        add(commit, `directory-reducer/${surface}`, module.directoryReducerPrompt(surface, true, true));
        add(commit, `directory-reducer/${surface}/depth-limit`, module.directoryReducerPrompt(surface, false, true));
        add(commit, `directory-reducer/${surface}/read-only`, module.directoryReducerPrompt(surface, true, false));
        add(commit, `directory-reducer/${surface}/read-only/depth-limit`, module.directoryReducerPrompt(surface, false, false));
      }
    }
  }
  // The text-channel prompt was first declared in its transport module, then moved into native/prompt.ts.
  // Keep those exact older texts as the same declared program piece instead of inferring scope from export names.
  const emulationRevisions = execFileSync('git', ['log', '--format=%H', '--', EMULATION_FILE], { cwd: REPO, encoding: 'utf8' }).split('\n').filter(Boolean);
  for (const commit of emulationRevisions) {
    const source = execFileSync('git', ['show', `${commit}:${EMULATION_FILE}`], { cwd: REPO, encoding: 'utf8', maxBuffer: 1 << 26 });
    const parsed = ts.createSourceFile(EMULATION_FILE, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    const importsAndReexports = parsed.statements.filter(statement => ts.isImportDeclaration(statement) ||
      (ts.isExportDeclaration(statement) && statement.moduleSpecifier !== undefined))
      .map(statement => [statement.getFullStart(), statement.getEnd()]);
    let isolated = source;
    for (const [start, end] of importsAndReexports.sort((a, b) => b[0] - a[0]))
      isolated = isolated.slice(0, start) + isolated.slice(end);
    const js = ts.transpileModule(isolated, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    const module = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
    add(commit, 'neuralese/text-emulation', module.TEXT_NEURALESE_EMULATION_PROMPT);
  }
  return [...byText.values()].sort((a, b) => a.id.localeCompare(b.id));
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
