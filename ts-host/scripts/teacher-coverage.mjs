import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';

export const digest = value => createHash('sha256').update(value).digest('hex');

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) =>
    a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}

export async function expandPatterns(patterns) {
  const found = new Set();
  for (const pattern of patterns) {
    if (!pattern.includes('*')) { found.add(pattern); continue; }
    if (pattern.includes('**') || /[*]/.test(dirname(pattern)))
      throw new Error(`unsupported coverage glob: ${pattern}`);
    const expression = new RegExp(`^${basename(pattern).split('*').map(part =>
      part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`);
    for (const name of await readdir(dirname(pattern))) if (expression.test(name))
      found.add(join(dirname(pattern), name));
  }
  return [...found].sort();
}

export async function buildSelection(paths, { perFamily, seed }) {
  const grouped = new Map(), identities = new Map(), sources = {}, added = new Set();
  for (const path of [...new Set(paths)].sort()) {
    const raw = await readFile(path);
    sources[path] = digest(raw);
    for (const line of raw.toString('utf8').split(/\r?\n/)) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      if (row.split !== 'train' || row.semantics?.contains_templates) continue;
      const revision = digest(canonical(row));
      const previous = identities.get(row.id);
      if (previous && previous !== revision) throw new Error(`conflicting revisions for program id ${row.id}`);
      identities.set(row.id, revision);
      if (added.has(row.id)) continue;
      added.add(row.id);
      const family = row.family ?? row.kind;
      if (!grouped.has(family)) grouped.set(family, []);
      grouped.get(family).push(row);
    }
  }
  const short = Object.fromEntries([...grouped].filter(([, rows]) => rows.length < perFamily)
    .map(([family, rows]) => [family, rows.length]));
  if (Object.keys(short).length) throw new Error(`families below ${perFamily} eligible programs: ${JSON.stringify(short)}`);
  const families = [...grouped].map(([family, rows]) => [family, rows.sort((a, b) =>
    Buffer.compare(createHash('sha256').update(`${seed}:${a.id}`).digest(),
      createHash('sha256').update(`${seed}:${b.id}`).digest())).slice(0, perFamily)])
    .sort(([a], [b]) => a.localeCompare(b));
  const selected = [];
  for (let variant = 0; variant < perFamily; variant++)
    for (const [, rows] of families) selected.push(rows[variant]);
  return { rows: selected, manifest: { schema: 'natlang.teacher_coverage_selection/1', seed,
    per_family: perFamily, families: grouped.size, programs: selected.length,
    sources, selected_ids: selected.map(row => row.id) } };
}

export async function auditCoverage(root, configPath, studioPath, programsPath) {
  const config = JSON.parse(await readFile(configPath, 'utf8'));
  const codebases = new Set();
  for (const item of await readdir(resolve(root, 'codebases'), { withFileTypes: true })) {
    if (!item.isDirectory()) continue;
    const files = await readdir(resolve(root, 'codebases', item.name));
    if (files.some(name => name.endsWith('.nl'))) codebases.add(`codebase:${item.name}`);
  }
  const applications = new Set((await readdir(resolve(root, 'applications')))
    .filter(name => name.endsWith('.py') || name.endsWith('.mjs')).map(name => `application:${name}`));
  const actual = new Set([...codebases, ...applications]);
  const owned = new Set(config.targets.flatMap(target => target.owns));
  const excluded = new Set(config.excluded ?? []);
  const unknown = [...actual].filter(item => !owned.has(item) && !excluded.has(item)).sort();
  const missing = [...owned].filter(item => !actual.has(item)).sort();
  const counts = new Map(), add = (key, split) => {
    const combined = `${key}\0${split}`; counts.set(combined, (counts.get(combined) ?? 0) + 1);
  };
  for (const line of (await readFile(studioPath, 'utf8')).split(/\r?\n/).filter(Boolean)) {
    const row = JSON.parse(line); add(row.target, row.split);
  }
  for (const line of (await readFile(programsPath, 'utf8')).split(/\r?\n/).filter(Boolean)) {
    const row = JSON.parse(line); add(`family:${row.family ?? row.kind}`, row.split ?? 'train');
  }
  const below = [];
  for (const target of config.targets) for (const [split, minimum] of Object.entries(config.minimums[target.provider])) {
    const found = counts.get(`${target.id}\0${split}`) ?? 0;
    if (found < minimum) below.push({ target: target.id, split, found, minimum });
  }
  return { schema: 'natlang.teacher_coverage_audit/1', discovered: actual.size,
    registered: owned.size, targets: config.targets.length, unknown, missing,
    below_minimum: below, ready: !(unknown.length || missing.length || below.length) };
}
