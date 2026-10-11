/** Reversible layouts for presenting records as a folder. */
import YAML from 'yaml';
import { Folder } from './scoped-fs.js';
const manifests = new WeakMap();
function safePart(value) {
    const part = String(value ?? '');
    if (!part || part === '.' || part === '..' || part.includes('/') || part.includes('\\') || part.includes('\0'))
        throw new RangeError(`invalid path component: ${JSON.stringify(part)}`);
    return part;
}
function pathFor(row, layout) {
    if (!layout.path)
        throw new TypeError('record layout needs path');
    const path = layout.path.replace(/\{([^{}]+)\}/g, (_, key) => safePart(row[key]));
    if (!path || path.startsWith('/') || path.split('/').some(part => !part || part === '.' || part === '..'))
        throw new RangeError(`invalid layout path: ${JSON.stringify(path)}`);
    return path;
}
function pathFields(path, layout) {
    if (!layout.path)
        return;
    const names = [];
    const expression = new RegExp(`^${layout.path.split(/(\{[^{}]+\})/g).map(part => {
        const field = /^\{([^{}]+)\}$/.exec(part);
        if (field) {
            names.push(field[1]);
            return '([^/]+)';
        }
        return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }).join('')}$`);
    const match = expression.exec(path);
    if (!match)
        return;
    return Object.fromEntries(names.map((name, index) => [name, match[index + 1]]));
}
function stableId(row, layout) {
    const id = safePart(row[layout.id]);
    return layout.disambiguation ? `${id}~${safePart(row[layout.disambiguation])}` : id;
}
function encode(row, layout, id) {
    const format = layout.format ?? 'frontmatter', bodyField = layout.body ?? 'body';
    if (format === 'json')
        return `${JSON.stringify({ ...row, [layout.id]: id }, null, 2)}\n`;
    if (format === 'text')
        return String(row[bodyField] ?? '');
    const fields = { ...row, [layout.id]: id };
    delete fields[bodyField];
    return `---\n${YAML.stringify(fields).trimEnd()}\n---\n${String(row[bodyField] ?? '')}`;
}
function decode(content, layout) {
    const format = layout.format ?? 'frontmatter', bodyField = layout.body ?? 'body';
    if (format === 'json') {
        const parsed = JSON.parse(content);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
            throw new Error('record JSON must be an object');
        return parsed;
    }
    if (format === 'text')
        return { [bodyField]: content };
    const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/.exec(content);
    if (!match)
        throw new Error('record is missing YAML front matter');
    const fields = YAML.parse(match[1]);
    if (!fields || typeof fields !== 'object' || Array.isArray(fields))
        throw new Error('record front matter must be a map');
    return { ...fields, [bodyField]: match[2] };
}
function csvCell(value) {
    const text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
function csvText(rows, columns) {
    return `${columns.map(csvCell).join(',')}\n${rows.map(row => columns.map(column => csvCell(row[column])).join(',')).join('\n')}${rows.length ? '\n' : ''}`;
}
function parseCsv(text) {
    const rows = [], row = [];
    let cell = '', quoted = false;
    for (let index = 0; index < text.length; index++) {
        const char = text[index];
        if (quoted) {
            if (char === '"' && text[index + 1] === '"') {
                cell += '"';
                index++;
            }
            else if (char === '"')
                quoted = false;
            else
                cell += char;
        }
        else if (char === '"' && !cell)
            quoted = true;
        else if (char === ',') {
            row.push(cell);
            cell = '';
        }
        else if (char === '\n') {
            row.push(cell.replace(/\r$/, ''));
            rows.push(row.splice(0));
            cell = '';
        }
        else
            cell += char;
    }
    if (quoted)
        throw new Error('unterminated quoted CSV field');
    if (cell || row.length) {
        row.push(cell);
        rows.push(row);
    }
    return rows;
}
function typedCell(value, example) {
    if (typeof example === 'number') {
        const number = Number(value);
        return value.trim() && Number.isFinite(number) ? number : value;
    }
    if (typeof example === 'boolean')
        return value === 'true' ? true : value === 'false' ? false : value;
    if (example && typeof example === 'object') {
        try {
            return JSON.parse(value);
        }
        catch {
            return value;
        }
    }
    return value;
}
/** Build a folder from records without asking callers to serialize files themselves. */
export function folderFromData(records, layout, access = 'write') {
    const tablePrimary = layout.writable === 'table';
    if (!layout.id || !tablePrimary && !layout.path || tablePrimary && !layout.table)
        throw new TypeError('layout needs id and either a record path or a writable table');
    if (layout.writable && !['records', 'table'].includes(layout.writable))
        throw new TypeError('invalid writable representation');
    const rows = new Map();
    const ids = new Map(), paths = new Set();
    const original = new Map();
    const tableRows = [];
    for (const row of records) {
        const snapshot = structuredClone(row);
        const id = stableId(snapshot, layout), path = tablePrimary ? '' : pathFor(snapshot, layout);
        if (ids.has(id))
            throw new Error(`duplicate record id: ${id}`);
        if (!tablePrimary && paths.has(path))
            throw new Error(`duplicate record path: ${path}`);
        ids.set(id, path);
        paths.add(path);
        original.set(id, snapshot);
        tableRows.push({ ...snapshot, [layout.id]: id });
        if (!tablePrimary)
            rows.set(path, { row: snapshot, id, size: new TextEncoder().encode(encode(snapshot, layout, id)).length });
    }
    const columns = [...new Set([layout.id, ...records.flatMap(row => Object.keys(row))])];
    if (layout.table) {
        if (paths.has(layout.table))
            throw new Error(`table collides with record: ${layout.table}`);
    }
    const source = {
        paths: () => rows.keys(),
        size: path => rows.get(path)?.size,
        get: path => {
            const entry = rows.get(path);
            return entry ? new TextEncoder().encode(encode(entry.row, layout, entry.id)) : undefined;
        },
    };
    const folder = tablePrimary ? Folder.fromFiles({ [layout.table]: csvText(tableRows, columns) }, access) :
        new Folder(source, access);
    if (layout.table && !tablePrimary) {
        folder.registerComputedFile(layout.table, current => {
            const rows = [];
            for (const path of current.filePaths()) {
                if (path === layout.table)
                    continue;
                let row;
                try {
                    row = decode(new TextDecoder().decode(current.readBytesSync(path)), layout);
                }
                catch {
                    continue;
                }
                const id = row[layout.id];
                if (typeof id !== 'string' || !ids.has(id))
                    continue;
                const location = pathFields(path, layout);
                if (location)
                    rows.push({ ...row, ...location, [layout.id]: id });
            }
            return new TextEncoder().encode(csvText(rows, columns));
        });
    }
    manifests.set(folder, { layout, ids, paths, table: layout.table, columns, original });
    return folder;
}
/** Recover edited records by embedded identity; moves become updated path fields. */
export async function folderToData(folder, layout) {
    const manifest = manifests.get(folder);
    const chosen = layout ?? manifest?.layout;
    if (!chosen)
        throw new TypeError('folderToData needs the layout used by Folder.fromData');
    const records = [], seen = new Set(), unknown = [], outputs = {};
    if (chosen.writable === 'table') {
        if (!chosen.table)
            throw new TypeError('table layout needs a table path');
        if (folder.isFile(chosen.table)) {
            const [header, ...body] = parseCsv(await folder.readText(chosen.table));
            if (!header?.includes(chosen.id) || new Set(header).size !== header.length)
                throw new Error(`CSV table needs unique headers including ${chosen.id}`);
            for (const [index, cells] of body.entries()) {
                if (cells.length !== header.length)
                    throw new Error(`CSV row ${index + 2} has ${cells.length} cells; expected ${header.length}`);
                const values = Object.fromEntries(header.map((name, column) => [name, cells[column]]));
                const id = values[chosen.id];
                if (!id || seen.has(id))
                    throw new Error(`duplicate or missing record id in CSV row ${index + 2}`);
                seen.add(id);
                const example = manifest?.original.get(id);
                records.push(Object.fromEntries(header.map(name => [name,
                    name === chosen.id ? id : typedCell(values[name], example?.[name])])));
            }
        }
        for (const entry of folder.listFiles())
            if (entry.path !== chosen.table) {
                unknown.push(entry.path);
                outputs[entry.path] = await folder.readText(entry.path);
            }
        return { records, deleted: [...manifest?.ids.keys() ?? []].filter(id => !seen.has(id)), unknown, outputs };
    }
    const originalByPath = new Map([...manifest?.ids ?? []].map(([id, path]) => [path, id]));
    for (const entry of folder.listFiles()) {
        const path = entry.path;
        if (path === chosen.table)
            continue;
        const content = await folder.readText(path);
        let row;
        try {
            row = decode(content, chosen);
        }
        catch {
            unknown.push(path);
            outputs[path] = content;
            continue;
        }
        const id = (chosen.format ?? 'frontmatter') === 'text' ? originalByPath.get(path) : row[chosen.id];
        if (typeof id !== 'string' || !manifest?.ids.has(id) || seen.has(id)) {
            unknown.push(path);
            outputs[path] = content;
            continue;
        }
        seen.add(id);
        const location = pathFields(path, chosen);
        if (!location) {
            unknown.push(path);
            outputs[path] = content;
            seen.delete(id);
            continue;
        }
        records.push({ ...row, ...location, [chosen.id]: id });
    }
    const deleted = [...manifest?.ids.keys() ?? []].filter(id => !seen.has(id));
    return { records, deleted, unknown, outputs };
}
