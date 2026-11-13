// "Download as natlang program": a Playground project as a folder the CLI runs (`natlang check`, `natlang run`,
// `natlang test`), packed in a zip written here (stored entries, no compression and no dependency).

const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc32 = bytes => { let c = 0xffffffff; for (const byte of bytes) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };

/** A zip archive of `files` (path → text) under the folder `folder`. */
export function zipFiles(folder, files) {
  const encoder = new TextEncoder(), local = [], central = [];
  let offset = 0;
  for (const [path, text] of Object.entries(files)) {
    const name = encoder.encode(`${folder}/${path}`), data = encoder.encode(text), crc = crc32(data);
    const header = (signature, extra) => { const view = new DataView(new ArrayBuffer(extra)); view.setUint32(0, signature, true); return view; };
    const entry = header(0x04034b50, 30);
    entry.setUint16(4, 20, true); entry.setUint16(6, 0x0800, true); // version, UTF-8 names; method 0 (stored), time 0
    entry.setUint32(14, crc, true); entry.setUint32(18, data.length, true); entry.setUint32(22, data.length, true); entry.setUint16(26, name.length, true);
    const record = header(0x02014b50, 46);
    record.setUint16(4, 20, true); record.setUint16(6, 20, true); record.setUint16(8, 0x0800, true);
    record.setUint32(16, crc, true); record.setUint32(20, data.length, true); record.setUint32(24, data.length, true);
    record.setUint16(28, name.length, true); record.setUint32(42, offset, true);
    local.push(new Uint8Array(entry.buffer), name, data); central.push(new Uint8Array(record.buffer), name);
    offset += 30 + name.length + data.length;
  }
  const size = central.reduce((sum, part) => sum + part.length, 0), end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, central.length / 2, true); end.setUint16(10, central.length / 2, true);
  end.setUint32(12, size, true); end.setUint32(16, offset, true);
  return new Blob([...local, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
}

const argumentType = value => typeof value === 'string' ? 'string' : typeof value === 'number' ? 'number' :
  typeof value === 'boolean' ? 'boolean' : Array.isArray(value) && value.every(item => typeof item === 'string') ? 'string[]' : 'unknown';

/**
 * The program folder of a Playground project: its files, plus natlang.json (a `.nl` root becomes the main target, its
 * inputs the arguments' defaults), examples.json (the expected result, when set) and package.json (ES modules).
 */
export function programFiles(project) {
  const name = project.name.toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+|-+$/g, '') || 'my-program';
  const natural = project.root.endsWith('.nl');
  const added = {
    'natlang.json': JSON.stringify({ schema: 'natlang.package/v2', name, version: '0.1.0', description: `${project.name}, from the natlang Playground.`,
      include: [...Object.keys(project.files), 'examples.json', 'package.json'].sort(),
      ...(natural ? { targets: { main: { entry: project.root, description: `Run ${project.root} on the Playground inputs, or supply your own.`,
        arguments: Object.entries(project.inputs).map(([key, value]) => ({ name: key, type: argumentType(value), default: value })) } } } : {}) }, null, 2) + '\n',
    'examples.json': JSON.stringify(natural && project.expected !== undefined ?
      [{ name: 'from the Playground', function: project.root, inputs: project.inputs, expected: project.expected }] : [], null, 2) + '\n',
    'package.json': JSON.stringify({ name, private: true, type: 'module' }, null, 2) + '\n',
    '.gitignore': '.natlang/\nnode_modules/\n*.d.nl.ts\n',
  };
  return { name, files: { ...project.files, ...Object.fromEntries(Object.entries(added).filter(([path]) => !Object.hasOwn(project.files, path))) } };
}
