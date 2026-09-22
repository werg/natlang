#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function countSteps(steps, operators) {
  if (!Array.isArray(steps)) return;
  for (const step of steps) {
    const operation = String(step?.op ?? 'unknown');
    operators.set(operation, (operators.get(operation) ?? 0) + 1);
    if (operation === 'branch') {
      countSteps(step.then, operators);
      countSteps(step.else, operators);
    }
  }
}

export function auditProgramRows(sources) {
  const ids = new Set(), groupSplits = new Map(), mix = new Map(), operators = new Map();
  for (const { path, rows } of sources) for (let index = 0; index < rows.length; index++) {
    const row = rows[index], at = `${path}:${index + 1}`;
    if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error(`${at}: record must be an object`);
    for (const field of ['id', 'source', 'split', 'kind', 'semantics'])
      if (!Object.hasOwn(row, field)) throw new Error(`${at}: missing ${field}`);
    if (ids.has(row.id)) throw new Error(`duplicate program: ${row.id}`);
    ids.add(row.id);
    if (!Array.isArray(row.source_groups)) throw new Error(`${at}: source_groups must be an array`);
    for (const group of row.source_groups) {
      const key = JSON.stringify([row.source, group]), prior = groupSplits.get(key);
      if (prior !== undefined && prior !== row.split)
        throw new Error(`source group crosses splits: ${row.source}/${group}`);
      groupSplits.set(key, row.split);
    }
    const mixKey = JSON.stringify([row.source, row.split, row.kind]);
    mix.set(mixKey, (mix.get(mixKey) ?? 0) + 1);
    const sem = row.semantics;
    if (row.kind === 'scene_program') for (const node of sem.nodes ?? [])
      operators.set(String(node.function), (operators.get(String(node.function)) ?? 0) + 1);
    else if (row.kind === 'numeric_program') for (const step of sem.steps ?? [])
      operators.set(String(step.op), (operators.get(String(step.op)) ?? 0) + 1);
    else if (row.kind === 'state_sequence')
      operators.set('state_transition', (operators.get('state_transition') ?? 0) + (sem.steps?.length ?? 0));
    else if (row.kind === 'lambda_source')
      operators.set(String(sem.operation), (operators.get(String(sem.operation)) ?? 0) + 1);
    else if (row.kind === 'proposal_review') {
      const key = `proposal:${sem.verdict}`; operators.set(key, (operators.get(key) ?? 0) + 1);
    } else if (row.kind === 'lambda_graph' || row.kind === 'lambda_scenario') {
      countSteps(sem.operations, operators);
      for (const steps of Object.values(sem.functions ?? {})) countSteps(steps, operators);
      if (sem.nested) {
        const key = `nested:${sem.nested.kind}`; operators.set(key, (operators.get(key) ?? 0) + 1);
      }
    } else operators.set('typed_decision', (operators.get('typed_decision') ?? 0) + (sem.decisions?.length ?? 0));
  }
  return { programs: ids.size, groups: groupSplits.size,
    mix: [...mix].map(([key, programs]) => { const [source, split, kind] = JSON.parse(key); return { source, split, kind, programs }; })
      .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
    operators: Object.fromEntries([...operators].sort(([a], [b]) => a.localeCompare(b))) };
}

async function readRows(path) {
  const text = await readFile(path, 'utf8');
  return text.split(/\r?\n/).flatMap((line, index) => {
    if (!line.trim()) return [];
    try { return [JSON.parse(line)]; }
    catch (error) { throw new Error(`${path}:${index + 1}: ${error.message}`); }
  });
}

async function main() {
  const args = process.argv.slice(2), outIndex = args.indexOf('--out');
  const output = outIndex < 0 ? null : args[outIndex + 1];
  const paths = args.filter((value, index) => index !== outIndex && index !== outIndex + 1 && !value.startsWith('--'));
  if (!paths.length || (outIndex >= 0 && !output))
    throw new Error('usage: audit-program-ir.mjs PATH... [--out REPORT.json]');
  const sources = await Promise.all(paths.map(async path => ({ path, rows: await readRows(path) })));
  const text = JSON.stringify(auditProgramRows(sources), null, 2) + '\n';
  if (output) { await mkdir(dirname(resolve(output)), { recursive: true }); await writeFile(output, text); }
  process.stdout.write(text);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
