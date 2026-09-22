#!/usr/bin/env node
/** Convert reviewed, admitted browser cases into the supported lambda_scenario IR subset. */
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, extname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadFunctionFile } from '../dist/native/source.js';
import { formatType } from '../dist/native/types.js';

export const PROGRAM_IR_VERSION = 'natlang.program/1';

function safeSourcePath(value) {
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0'))
    throw new Error(`unsafe or unsupported source path: ${value}`);
  const parts = value.split('/');
  if (isAbsolute(value) || parts.some(part => !part || part === '.' || part === '..') ||
      !['.nl', '.ts'].includes(extname(value)))
    throw new Error(`unsafe or unsupported source path: ${value}`);
  return parts;
}

async function rootLambda(caseRecord) {
  const source = caseRecord.source ?? {}, files = source.files ?? {};
  const rootPath = source.root;
  const parts = safeSourcePath(rootPath);
  if (extname(rootPath) !== '.nl' || typeof files[rootPath] !== 'string')
    throw new Error('only natural-language root functions can become lambda scenarios');
  const folder = await mkdtemp(join(tmpdir(), 'natlang-case-'));
  try {
    for (const [name, body] of Object.entries(files)) {
      const fileParts = safeSourcePath(name);
      if (typeof body !== 'string') throw new Error('source file must contain text');
      const target = join(folder, ...fileParts);
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, body);
    }
    const functionNode = loadFunctionFile(join(folder, ...parts));
    if (functionNode.kind !== 'instructions' || Object.keys(functionNode.codebase).length)
      throw new Error('linked child functions need a graph adapter');
    return { $lambda: {
      type: formatType(functionNode.type), instructions: functionNode.body,
      ...(Object.keys(functionNode.typesSrc).length ? { types: functionNode.typesSrc } : {}),
      ...(functionNode.effects.length ? { effects: functionNode.effects } : {}),
      args: caseRecord.inputs ?? {},
    } };
  } finally { await rm(folder, { recursive: true, force: true }); }
}

function operations(events) {
  const actions = events.filter(event => event.kind === 'action');
  if (!actions.length) throw new Error('no recorded actions');
  return actions.map(action => {
    if (action.call_id !== '$root@1' || action.outcome !== 'ok')
      throw new Error('nested or rejected actions need a richer adapter');
    const args = action.arguments ?? {};
    if (action.name === 'write' && typeof args.path === 'string' && typeof args.type === 'string') {
      const result = { op: 'assign', target: args.path, value_type: args.type };
      if (Object.hasOwn(args, 'source')) result.from = args.source;
      else if (Object.hasOwn(args, 'value')) result.value = args.value;
      else throw new Error('write action has neither source nor value');
      if (Object.hasOwn(args, 'done')) result.completion = args.done;
      return result;
    }
    if (action.name === 'report_error' && typeof args.message === 'string')
      return { op: 'fail', message: args.message };
    if (action.name === 'report_blocker' && typeof args.missing === 'string')
      return { op: 'block', missing: args.missing };
    throw new Error(`action ${JSON.stringify(action.name)} needs a richer adapter`);
  });
}

export async function convertCase(caseRecord) {
  if (caseRecord?.schema !== 'natlang.playground.case/1' || caseRecord.reviewStatus !== 'accepted' ||
      caseRecord.admission?.admitted !== true) throw new Error('case is not reviewed and exactly admitted');
  const events = caseRecord.trace ?? [];
  if (!events.length || events[0].kind !== 'manifest' || events.some((event, index) =>
    event.seq !== index || event.version !== 'reduction-trace/1'))
    throw new Error('missing or discontinuous reduction trace');
  const finalStates = events.filter(event => event.kind === 'state' && event.phase === 'final');
  if (finalStates.length !== 1) throw new Error('trace needs one final state');
  const expected = caseRecord.expected ?? {}, final = finalStates[0];
  if (expected.kind !== final.outcome || (expected.kind === 'done' && expected.value !== final.value))
    throw new Error('expected result differs from recorded outcome');
  if (events.some(event => event.kind === 'effect')) throw new Error('host effects need a richer adapter');
  const semanticActions = events.filter(event => event.kind === 'action');
  const contract = { kind: expected.kind, ...(Object.hasOwn(expected, 'value') ? { value: expected.value } : {}),
    effects: caseRecord.effects ?? [],
    required_actions: semanticActions.map(event => ({ tool: event.name, arguments: event.arguments })),
    instruction_obligations: caseRecord.requiredActions ?? [],
    constrained_calls: caseRecord.constrainedCalls ?? [] };
  if (['error', 'blocked'].includes(expected.kind))
    contract.explanation = Object.values(semanticActions.at(-1)?.arguments ?? {})[0] ?? '';
  const id = String(caseRecord.id);
  return { version: PROGRAM_IR_VERSION, id, kind: 'lambda_scenario', source: 'browser_playground',
    split: caseRecord.split ?? 'train', source_ids: [id],
    source_groups: [String(caseRecord.groupId ?? caseRecord.projectId ?? id)],
    source_revisions: [String(caseRecord.revision)], license: 'user-authored',
    gold_sources: ['reviewed-browser-case'],
    semantics: { root: await rootLambda(caseRecord), operations: operations(events), contract } };
}

export async function importCases(sourcePath, outputPath, rejectsPath = null) {
  const source = resolve(sourcePath), output = resolve(outputPath);
  if (source === output) throw new Error('input and output must differ');
  const rejectFile = resolve(rejectsPath ?? join(dirname(output), `${basename(output, extname(output))}-rejects.jsonl`));
  await mkdir(dirname(output), { recursive: true });
  const accepted = [], rejected = [];
  const lines = (await readFile(source, 'utf8')).split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    if (!lines[index].trim()) continue;
    let record = null;
    try {
      record = JSON.parse(lines[index]);
      accepted.push(await convertCase(record));
    } catch (error) {
      rejected.push({ line: index + 1, id: record && typeof record === 'object' ? record.id ?? null : null,
        reason: error instanceof Error ? error.message : String(error) });
    }
  }
  await writeFile(output, accepted.map(row => JSON.stringify(row)).join('\n') + (accepted.length ? '\n' : ''));
  await writeFile(rejectFile, rejected.map(row => JSON.stringify(row)).join('\n') + (rejected.length ? '\n' : ''));
  console.log(JSON.stringify({ accepted: accepted.length, rejected: rejected.length,
    program_ir: output, rejects: rejectFile }));
  return { accepted: accepted.length, rejected: rejected.length, output, rejects: rejectFile };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [, , source, output, ...rest] = process.argv;
  if (!source || !output || (rest.length && (rest.length !== 2 || rest[0] !== '--rejects'))) {
    console.error('usage: node scripts/import-playground-cases.mjs INPUT.jsonl OUTPUT.jsonl [--rejects FILE]');
    process.exitCode = 2;
  } else {
    const rejectAt = rest[0] === '--rejects' ? process.argv.indexOf('--rejects') + 1 : -1;
    const rejects = rejectAt > 0 ? process.argv[rejectAt] : null;
    if (rest.length && !rejects) { console.error('unknown option'); process.exitCode = 2; }
    else importCases(source, output, rejects).catch(error => {
      console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1;
    });
  }
}
