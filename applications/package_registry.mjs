/** Offline, content-addressed natlang package registry and installer. */
import { createHash, randomUUID } from 'node:crypto';
import { lstat, mkdir, readFile, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { NativeSourceWorkspace } from '../ts-host/dist/index.js';

const namePattern = /^[a-z][a-z0-9_]*$/;
const versionPattern = /^(\d+)\.(\d+)\.(\d+)$/;
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const canonical = value => Array.isArray(value) ? value.map(canonical) :
  value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) =>
    a.localeCompare(b)).map(([key, item]) => [key, canonical(item)])) : value;
const digest = value => hash(canonical(value));

function version(value) {
  const match = versionPattern.exec(value);
  if (!match) throw new Error(`unsupported version: ${value}`);
  return match.slice(1).map(Number);
}
function compare(a, b) {
  const x = version(a), y = version(b);
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
}
function satisfies(value, range) {
  if (range === '*') return true;
  if (versionPattern.test(range)) return value === range;
  if (!range.startsWith('^')) throw new Error(`unsupported constraint: ${range}`);
  const min = version(range.slice(1)), actual = version(value);
  const upper = min[0] > 0 ? [min[0] + 1, 0, 0] : min[1] > 0 ?
    [0, min[1] + 1, 0] : [0, 0, min[2] + 1];
  const cmp = (a, b) => { for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] - b[i]; return 0; };
  return cmp(actual, min) >= 0 && cmp(actual, upper) < 0;
}

export class PackageRegistry {
  constructor(packages, installRoot, { maxSolutions = null } = {}) {
    this.installRoot = resolve(installRoot);
    this.maxSolutions = maxSolutions;
    this.index = new Map();
    this.events = [];
    for (const supplied of packages) {
      const manifest = structuredClone(supplied);
      if (!namePattern.test(manifest.name) || !versionPattern.test(manifest.version) ||
          !Array.isArray(manifest.engines) || !manifest.engines.length ||
          !manifest.definitions || typeof manifest.definitions !== 'object')
        throw new Error('invalid package manifest');
      if (Object.keys(manifest.definitions).some(name => !namePattern.test(name)))
        throw new Error(`invalid export in ${manifest.name}`);
      if (Object.entries(manifest.dependencies ?? {}).some(([name, range]) =>
        !namePattern.test(name) || typeof range !== 'string'))
        throw new Error(`invalid dependency in ${manifest.name}`);
      for (const engine of manifest.engines)
        if (!['typescript-host', 'quickjs-isolated'].includes(engine))
          throw new Error(`unknown engine: ${engine}`);
      const id = `${manifest.name}@${manifest.version}`;
      if (this.index.has(id)) throw new Error(`duplicate package: ${id}`);
      this.index.set(id, { manifest, sha256: digest(manifest) });
    }
  }

  catalog(name) {
    return [...this.index.entries()].filter(([, row]) => !name || row.manifest.name === name)
      .map(([id, row]) => ({ id, name: row.manifest.name, version: row.manifest.version,
        description: String(row.manifest.description ?? ''), sha256: row.sha256,
        engines: row.manifest.engines })).sort((a, b) => a.name.localeCompare(b.name) ||
          compare(b.version, a.version));
  }

  solutions(request) {
    if (!namePattern.test(request.name) || !['typescript-host', 'quickjs-isolated'].includes(request.engine))
      throw new Error('invalid root package or engine');
    const solutions = [];
    const visit = (name, range, selected, active) => {
      if (active.includes(name)) return [];
      const chosen = selected.get(name);
      if (chosen) return satisfies(chosen.manifest.version, range) ? [selected] : [];
      const candidates = [...this.index.values()].filter(row => row.manifest.name === name &&
        row.manifest.engines.includes(request.engine) && satisfies(row.manifest.version, range))
        .sort((a, b) => compare(b.manifest.version, a.manifest.version));
      const out = [];
      for (const candidate of candidates) {
        let branches = [new Map([...selected, [name, candidate]])];
        for (const [dependency, constraint] of Object.entries(candidate.manifest.dependencies ?? {}).sort()) {
          branches = branches.flatMap(branch => visit(dependency, constraint, branch, [...active, name]));
          if (!branches.length) break;
        }
        out.push(...branches);
        if (this.maxSolutions !== null && out.length > this.maxSolutions)
          throw new Error('solution set exceeds configured host allowance');
      }
      return out;
    };
    for (const selected of visit(request.name, request.range, new Map(), [])) {
      const pins = [...selected.values()].map(row => ({ name: row.manifest.name,
        version: row.manifest.version, sha256: row.sha256 })).sort((a, b) => a.name.localeCompare(b.name));
      const lock = { root: request.name, engine: request.engine, packages: pins };
      solutions.push({ ...lock, id: digest(lock) });
    }
    this.events.push({ operation: 'packages.solve', name: request.name, range: request.range,
      engine: request.engine, count: solutions.length });
    return solutions;
  }

  verify(lock) {
    const expected = this.solutions({ name: lock.root,
      range: lock.packages.find(pin => pin.name === lock.root)?.version ?? '', engine: lock.engine });
    if (!expected.some(item => item.id === lock.id && digest({ root: lock.root,
      engine: lock.engine, packages: lock.packages }) === lock.id)) throw new Error('lock does not match registry');
    return lock.packages.map(pin => {
      const row = this.index.get(`${pin.name}@${pin.version}`);
      if (!row || digest(row.manifest) !== pin.sha256) throw new Error(`package content changed: ${pin.name}`);
      return row.manifest;
    });
  }

  bundle(lock) {
    const manifests = this.verify(lock);
    const definitions = {};
    for (const manifest of manifests) for (const [name, def] of Object.entries(manifest.definitions)) {
      const qualified = `${manifest.name}__${name}`;
      if (definitions[qualified]) throw new Error(`duplicate export: ${qualified}`);
      const uses = Object.fromEntries(Object.entries(def.uses ?? {}).map(([alias, target]) => {
        const [packageName, exportName] = target.includes('/') ? target.split('/') : [manifest.name, target];
        return [alias, `${packageName}__${exportName}`];
      }));
      definitions[qualified] = { ...def, uses };
      if (def.code && !manifest.engines.includes(def.engine ?? 'quickjs-isolated'))
        throw new Error(`undeclared engine in ${qualified}`);
    }
    const root = `${lock.root}__main`;
    const workspace = new NativeSourceWorkspace(definitions, root);
    return { schema: 'natlang-package-bundle/v1', lock, root, definitions,
      revision: workspace.revision };
  }

  async install(lock, target) {
    if (!namePattern.test(target)) throw new Error('invalid install target');
    const bundle = this.bundle(lock);
    await mkdir(this.installRoot, { recursive: true });
    const root = await realpath(this.installRoot);
    const destination = join(root, target);
    try { await lstat(destination); throw new Error(`install target already exists: ${target}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const stage = join(root, `.store-${randomUUID()}`);
    await mkdir(stage);
    try {
      await writeFile(join(stage, 'bundle.json'), JSON.stringify(bundle, null, 2));
      await writeFile(join(stage, 'lock.json'), JSON.stringify(lock, null, 2));
      await symlink(stage, destination, 'dir');
    } catch (error) {
      await rm(stage, { recursive: true, force: true });
      throw error;
    }
    this.events.push({ operation: 'packages.install', target, lock_id: lock.id,
      revision: bundle.revision });
    return { status: 'installed', target, revision: bundle.revision, detail: '' };
  }

  async load(target) {
    if (!namePattern.test(target)) throw new Error('invalid install target');
    const path = await realpath(join(this.installRoot, target));
    if (resolve(path, '..') !== await realpath(this.installRoot)) throw new Error('install path escaped root');
    const bundle = JSON.parse(await readFile(join(path, 'bundle.json'), 'utf8'));
    const expected = this.bundle(bundle.lock);
    if (digest(bundle) !== digest(expected)) throw new Error('installed bundle changed');
    return new NativeSourceWorkspace(bundle.definitions, bundle.root);
  }

  drainEvents() { return this.events.splice(0); }
}
