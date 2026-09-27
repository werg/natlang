/** Reversible layouts for presenting records as a folder. */
import YAML from 'yaml';
import { Folder, type FolderAccess, type FolderSource } from './scoped-fs.js';

export type FolderDataLayout = {
  /** The unique record key. */
  id: string;
  /** A relative path template such as `emails/{group}/{id}.md`. */
  path?: string;
  /** `frontmatter` preserves typed fields and free text; `json` and `text` are also supported. */
  format?: 'frontmatter' | 'json' | 'text';
  /** Field containing the free-text body. Defaults to `body`. */
  body?: string;
  /** An optional deterministic field appended when id values are duplicated. */
  disambiguation?: string;
  /** The one writable representation. Defaults to per-record files. */
  writable?: 'records' | 'table';
  /** CSV path: a regenerated view for records, or the writable source in table mode. */
  table?: string;
};

export type FolderDataResult = {
  records: Record<string, unknown>[];
  deleted: string[];
  unknown: string[];
  /** Non-record files written by a reducer, such as a report. */
  outputs: Record<string, string>;
};

type Manifest = { layout: FolderDataLayout; ids: Map<string, string>; paths: Set<string>;
  table?: string; columns: string[]; original: Map<string, Record<string, unknown>> };
const manifests = new WeakMap<Folder, Manifest>();

function safePart(value: unknown): string {
  const part = String(value ?? '');
  if (!part || part === '.' || part === '..' || part.includes('/') || part.includes('\\') || part.includes('\0'))
    throw new RangeError(`invalid path component: ${JSON.stringify(part)}`);
  return part;
}
function pathFor(row: Record<string, unknown>, layout: FolderDataLayout): string {
  if (!layout.path) throw new TypeError('record layout needs path');
  const path = layout.path.replace(/\{([^{}]+)\}/g, (_, key: string) => safePart(row[key]));
  if (!path || path.startsWith('/') || path.split('/').some(part => !part || part === '.' || part === '..'))
    throw new RangeError(`invalid layout path: ${JSON.stringify(path)}`);
  return path;
}
function pathFields(path: string, layout: FolderDataLayout): Record<string, string> | undefined {
  if (!layout.path) return;
  const names: string[] = [];
  const expression = new RegExp(`^${layout.path.split(/(\{[^{}]+\})/g).map(part => {
    const field = /^\{([^{}]+)\}$/.exec(part);
    if (field) { names.push(field[1]!); return '([^/]+)'; }
    return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }).join('')}$`);
  const match = expression.exec(path);
  if (!match) return;
  return Object.fromEntries(names.map((name, index) => [name, match[index + 1]!]));
}
function stableId(row: Record<string, unknown>, layout: FolderDataLayout): string {
  const id = safePart(row[layout.id]);
  return layout.disambiguation ? `${id}~${safePart(row[layout.disambiguation])}` : id;
}
function encode(row: Record<string, unknown>, layout: FolderDataLayout, id: string): string {
  const format = layout.format ?? 'frontmatter', bodyField = layout.body ?? 'body';
  if (format === 'json') return `${JSON.stringify({ ...row, [layout.id]: id }, null, 2)}\n`;
  if (format === 'text') return String(row[bodyField] ?? '');
  const fields = { ...row, [layout.id]: id };
  delete fields[bodyField];
  return `---\n${YAML.stringify(fields).trimEnd()}\n---\n${String(row[bodyField] ?? '')}`;
}
function decode(content: string, layout: FolderDataLayout): Record<string, unknown> {
  const format = layout.format ?? 'frontmatter', bodyField = layout.body ?? 'body';
  if (format === 'json') {
    const parsed: unknown = JSON.parse(content);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('record JSON must be an object');
    return parsed as Record<string, unknown>;
  }
  if (format === 'text') return { [bodyField]: content };
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/.exec(content);
  if (!match) throw new Error('record is missing YAML front matter');
  const fields: unknown = YAML.parse(match[1]!);
  if (!fields || typeof fields !== 'object' || Array.isArray(fields)) throw new Error('record front matter must be a map');
  return { ...(fields as Record<string, unknown>), [bodyField]: match[2]! };
}
function csvCell(value: unknown): string {
  const text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
function csvText(rows: Record<string, unknown>[], columns: string[]): string {
  return `${columns.map(csvCell).join(',')}\n${rows.map(row =>
    columns.map(column => csvCell(row[column])).join(',')).join('\n')}${rows.length ? '\n' : ''}`;
}
function parseCsv(text: string): string[][] {
  const rows: string[][] = [], row: string[] = [];
  let cell = '', quoted = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!;
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { cell += '"'; index++; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"' && !cell) quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row.splice(0)); cell = ''; }
    else cell += char;
  }
  if (quoted) throw new Error('unterminated quoted CSV field');
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
function typedCell(value: string, example: unknown): unknown {
  if (typeof example === 'number') { const number = Number(value); return value.trim() && Number.isFinite(number) ? number : value; }
  if (typeof example === 'boolean') return value === 'true' ? true : value === 'false' ? false : value;
  if (example && typeof example === 'object') {
    try { return JSON.parse(value) as unknown; } catch { return value; }
  }
  return value;
}

/** Build a folder from records without asking callers to serialize files themselves. */
export function folderFromData(records: Record<string, unknown>[], layout: FolderDataLayout,
  access: FolderAccess = 'write'): Folder {
  const tablePrimary = layout.writable === 'table';
  if (!layout.id || !tablePrimary && !layout.path || tablePrimary && !layout.table)
    throw new TypeError('layout needs id and either a record path or a writable table');
  if (layout.writable && !['records', 'table'].includes(layout.writable)) throw new TypeError('invalid writable representation');
  const rows = new Map<string, { row: Record<string, unknown>; id: string; size: number }>();
  const ids = new Map<string, string>(), paths = new Set<string>();
  const original = new Map<string, Record<string, unknown>>();
  const tableRows: Record<string, unknown>[] = [];
  for (const row of records) {
    const snapshot = structuredClone(row);
    const id = stableId(snapshot, layout), path = tablePrimary ? '' : pathFor(snapshot, layout);
    if (ids.has(id)) throw new Error(`duplicate record id: ${id}`);
    if (!tablePrimary && paths.has(path)) throw new Error(`duplicate record path: ${path}`);
    ids.set(id, path); paths.add(path);
    original.set(id, snapshot);
    tableRows.push({ ...snapshot, [layout.id]: id });
    if (!tablePrimary)
      rows.set(path, { row: snapshot, id, size: new TextEncoder().encode(encode(snapshot, layout, id)).length });
  }
  const columns = [...new Set([layout.id, ...records.flatMap(row => Object.keys(row))])];
  if (layout.table) {
    if (paths.has(layout.table)) throw new Error(`table collides with record: ${layout.table}`);
  }
  const source: FolderSource = {
    paths: () => rows.keys(),
    size: path => rows.get(path)?.size,
    get: path => {
      const entry = rows.get(path);
      return entry ? new TextEncoder().encode(encode(entry.row, layout, entry.id)) : undefined;
    },
  };
  const folder = tablePrimary ? Folder.fromFiles({ [layout.table!]: csvText(tableRows, columns) }, access) :
    new Folder(source, access);
  if (layout.table && !tablePrimary) {
    folder.registerComputedFile(layout.table, current => {
      const rows: Record<string, unknown>[] = [];
      for (const path of current.filePaths()) {
        if (path === layout.table) continue;
        let row: Record<string, unknown>;
        try { row = decode(new TextDecoder().decode(current.readBytesSync(path)), layout); }
        catch { continue; }
        const id = row[layout.id];
        if (typeof id !== 'string' || !ids.has(id)) continue;
        const location = pathFields(path, layout);
        if (location) rows.push({ ...row, ...location, [layout.id]: id });
      }
      return new TextEncoder().encode(csvText(rows, columns));
    });
  }
  manifests.set(folder, { layout, ids, paths, table: layout.table, columns, original });
  return folder;
}

/** Recover edited records by embedded identity; moves become updated path fields. */
export async function folderToData(folder: Folder, layout?: FolderDataLayout): Promise<FolderDataResult> {
  const manifest = manifests.get(folder);
  const chosen = layout ?? manifest?.layout;
  if (!chosen) throw new TypeError('folderToData needs the layout used by Folder.fromData');
  const records: Record<string, unknown>[] = [], seen = new Set<string>(), unknown: string[] = [], outputs: Record<string, string> = {};
  if (chosen.writable === 'table') {
    if (!chosen.table) throw new TypeError('table layout needs a table path');
    if (folder.isFile(chosen.table)) {
      const [header, ...body] = parseCsv(await folder.readText(chosen.table));
      if (!header?.includes(chosen.id) || new Set(header).size !== header.length)
        throw new Error(`CSV table needs unique headers including ${chosen.id}`);
      for (const [index, cells] of body.entries()) {
        if (cells.length !== header.length) throw new Error(`CSV row ${index + 2} has ${cells.length} cells; expected ${header.length}`);
        const values = Object.fromEntries(header.map((name, column) => [name, cells[column]!]));
        const id = values[chosen.id];
        if (!id || seen.has(id)) throw new Error(`duplicate or missing record id in CSV row ${index + 2}`);
        seen.add(id);
        const example = manifest?.original.get(id);
        records.push(Object.fromEntries(header.map(name => [name,
          name === chosen.id ? id : typedCell(values[name]!, example?.[name])])));
      }
    }
    for (const entry of folder.listFiles()) if (entry.path !== chosen.table) {
      unknown.push(entry.path); outputs[entry.path] = await folder.readText(entry.path);
    }
    return { records, deleted: [...manifest?.ids.keys() ?? []].filter(id => !seen.has(id)), unknown, outputs };
  }
  const originalByPath = new Map([...manifest?.ids ?? []].map(([id, path]) => [path, id]));
  for (const entry of folder.listFiles()) {
    const path = entry.path;
    if (path === chosen.table) continue;
    const content = await folder.readText(path);
    let row: Record<string, unknown>;
    try { row = decode(content, chosen); }
    catch { unknown.push(path); outputs[path] = content; continue; }
    const id = (chosen.format ?? 'frontmatter') === 'text' ? originalByPath.get(path) : row[chosen.id];
    if (typeof id !== 'string' || !manifest?.ids.has(id) || seen.has(id)) {
      unknown.push(path); outputs[path] = content; continue;
    }
    seen.add(id);
    const location = pathFields(path, chosen);
    if (!location) { unknown.push(path); outputs[path] = content; seen.delete(id); continue; }
    records.push({ ...row, ...location, [chosen.id]: id });
  }
  const deleted = [...manifest?.ids.keys() ?? []].filter(id => !seen.has(id));
  return { records, deleted, unknown, outputs };
}
