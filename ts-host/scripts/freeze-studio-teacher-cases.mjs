#!/usr/bin/env node
/** Freeze varied Studio interactions and exact host outcomes for teacher collection. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { apps } from '../studio/apps/index.mjs';
import { operationFields } from '../studio/shared/contracts.mjs';
import { applyOperation } from '../studio/shared/host.mjs';
import { sourceNames } from '../studio/shared/program.mjs';
import { simulateInventory } from '../studio/apps/worlds.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const stable = value => JSON.stringify(value, Object.keys(value ?? {}).sort());
const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');

const services = {
  trial: config => simulateInventory(config),
  readValue: async () => [2, 3, 5, 7, 11],
  cell: async cell => ({ id: `value-${cell.id}`, preview: cell.id === 'numbers' ? '[2,3,5,7,11]' : '28' }),
  runSource: async () => ({ value: 'fixture', trace_id: 'frozen-trace', trace_count: 1 }),
  inspectTrace: async () => 'frozen trace frame',
  service: async operation => {
    if (operation === 'media.sample') return { asset: 'asset-fixture.mp4' };
    if (operation === 'media.transform') return { asset: 'asset-transformed.mp4' };
    if (operation === 'build.run') return { status: 'ok', output: 'FIXTURE' };
    if (operation === 'terminal.run') return { status: 'succeeded', output: 'fixture stdout' };
    if (operation === 'packages.catalog') return { packages: [{ name: 'greetings', versions: ['1.0.0', '1.1.0'] }] };
    if (operation === 'packages.resolve') return { locks: [{ id: 'fixture-lock' }] };
    if (operation === 'packages.install') return { status: 'installed', id: 'fixture-install' };
    if (operation === 'repository.check') return { status: 'reviewable', checks: [] };
    throw new Error(`No frozen service for ${operation}`);
  },
};

function fieldsOf(form) {
  return Object.fromEntries((form?.fields ?? []).map(field => [field.name, field.value]));
}

function controls(spec) {
  const out = [{ ...spec.smoke }];
  for (const panel of spec.panels(spec.initial())) {
    const groups = [panel, ...(panel.extraForms ?? [])];
    for (const form of groups) for (const action of form.actions ?? [])
      out.push({ action: action.kind, ...fieldsOf(form), ...(action.data ?? {}) });
    for (const action of panel.rowActions ?? []) {
      const rows = panel.rows ?? [];
      for (const row of rows.slice(0, 2)) out.push({ action: action.kind, ...fieldsOf(panel),
        target: row.id, ...(action.data ?? {}) });
    }
  }
  return out;
}

function variants(spec, candidate) {
  const fields = operationFields[spec.id]?.[candidate.action] ?? [];
  const out = [candidate];
  for (const field of fields) {
    const key = field.replace('?', '');
    if (!Object.hasOwn(candidate, key)) continue;
    const value = candidate[key];
    if (typeof value === 'number') out.push({ ...candidate, [key]: value + 1 });
    else if (typeof value === 'string' && value) out.push({ ...candidate, [key]: value + ' · variant' });
    else if (Array.isArray(value)) out.push({ ...candidate, [key]: [...value].reverse() });
  }
  return out;
}

async function sourceRevision(spec) {
  const files = {};
  for (const name of sourceNames) {
    const path = resolve(root, 'ts-host', 'studio', 'programs', spec.id, name);
    files[relative(root, path)] = await readFile(path, 'utf8');
  }
  return digest({ files, decisionType: spec.decisionType, initial: spec.initial() });
}

export async function freeze({ minimum = 6 } = {}) {
  const rows = [];
  for (const spec of apps) {
    const seen = new Set(), candidates = [];
    for (const base of controls(spec)) for (const event of variants(spec, base)) {
      const key = stable(event);
      if (!seen.has(key)) { seen.add(key); candidates.push(event); }
    }
    for (let ordinal = 1; candidates.length < minimum; ordinal++) {
      for (const base of controls(spec)) {
        const event = { ...base };
        const keys = Object.keys(event).filter(name => name !== 'action' && typeof event[name] === 'string');
        if (!keys.length) continue;
        for (const key of keys) event[key] += ` · case ${ordinal}`;
        const identity = stable(event);
        if (!seen.has(identity)) { seen.add(identity); candidates.push(event); }
        if (candidates.length >= minimum) break;
      }
    }
    // Some compact apps expose few controls. Deterministically vary the smoke
    // payload so every target still has enough positive and negative evidence.
    for (let ordinal = 1; candidates.length < minimum; ordinal++) {
      const event = { ...spec.smoke };
      const key = Object.keys(event).find(name => typeof event[name] === 'string' && name !== 'action');
      if (key) event[key] += ` · case ${ordinal}`;
      else event.__case = ordinal; // last resort for a truly argument-free application
      const identity = stable(event);
      if (!seen.has(identity)) { seen.add(identity); candidates.push(event); }
    }
    const revision = await sourceRevision(spec);
    for (const [ordinal, decision] of candidates.slice(0, minimum).entries()) {
      const before = spec.initial();
      const outcome = await applyOperation(spec, before, decision, services);
      rows.push({ schema: 'natlang.studio_teacher_case/1',
        id: `studio:${spec.id}:${digest(decision).slice(0, 16)}`, target: `studio:${spec.id}`,
        project: spec.project, split: ordinal === minimum - 1 ? 'eval' : 'train',
        source_revision: revision, initial_state: before,
        event: { id: `teacher-${ordinal}`, kind: decision.action, value: JSON.stringify(decision) },
        decision, expected: outcome });
    }
  }
  return rows;
}

async function main() {
  const output = process.argv[2];
  const minimum = Number(process.argv[3] ?? 6);
  if (!output || !Number.isInteger(minimum) || minimum < 2)
    throw new Error('usage: freeze-studio-teacher-cases.mjs OUTPUT.jsonl [MINIMUM>=2]');
  const rows = await freeze({ minimum });
  const path = resolve(output), building = path + '.building';
  await mkdir(dirname(path), { recursive: true });
  await writeFile(building, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
  await rename(building, path);
  const counts = Object.fromEntries(apps.map(spec => [spec.id, rows.filter(row => row.target === `studio:${spec.id}`).length]));
  await writeFile(path + '.manifest.json', JSON.stringify({ schema: 'natlang.teacher_coverage/1',
    minimum, targets: counts, cases: rows.length, sha256: digest(await readFile(path)) }, null, 2) + '\n');
  console.log(`${rows.length} frozen cases across ${apps.length} Studio applications -> ${path}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
