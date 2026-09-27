/** Bring ZIP and TAR workspaces into the same Folder surface. */
import { gunzipSync, unzipSync } from 'fflate';
import { Folder, type FolderAccess } from './scoped-fs.js';

export type ArchiveFormat = 'zip' | 'tar';
const decoder = new TextDecoder();
function cleanArchivePath(path: string): string | undefined {
  const normalized = path.replace(/^\.\//, '').replace(/\/$/, '');
  if (!normalized || path.endsWith('/')) return;
  if (normalized.startsWith('/') || normalized.includes('\\') || normalized.includes('\0') ||
      normalized.split('/').some(part => !part || part === '.' || part === '..'))
    throw new RangeError(`archive entry escapes folder: ${path}`);
  return normalized;
}
function tarFiles(data: Uint8Array): Record<string, Uint8Array> {
  const files: Record<string, Uint8Array> = {};
  for (let offset = 0; offset + 512 <= data.length;) {
    const header = data.subarray(offset, offset + 512);
    if (header.every(byte => byte === 0)) break;
    const field = (start: number, length: number) => decoder.decode(header.subarray(start, start + length)).replace(/\0.*$/s, '').trim();
    const name = field(0, 100), prefix = field(345, 155), path = prefix ? `${prefix}/${name}` : name;
    const size = Number.parseInt(field(124, 12).replace(/\s/g, '') || '0', 8);
    if (!Number.isSafeInteger(size) || size < 0 || offset + 512 + size > data.length)
      throw new Error(`invalid TAR entry: ${path}`);
    const type = header[156];
    if (type === 0 || type === 48) {
      const clean = cleanArchivePath(path);
      if (clean) {
        if (Object.hasOwn(files, clean)) throw new Error(`duplicate archive entry: ${clean}`);
        files[clean] = data.slice(offset + 512, offset + 512 + size);
      }
    }
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  return files;
}

/** Open archive bytes as an in-memory folder, in Node or a browser worker. */
export function openArchive(input: Uint8Array | ArrayBuffer, format?: ArchiveFormat,
  access: FolderAccess = 'read'): Folder {
  const data = input instanceof Uint8Array ? input : new Uint8Array(input);
  const kind = format ?? (data[0] === 0x50 && data[1] === 0x4b ? 'zip' : 'tar');
  if (kind === 'zip') {
    const extracted = unzipSync(data), files: Record<string, Uint8Array> = {};
    for (const [path, contents] of Object.entries(extracted)) {
      const clean = cleanArchivePath(path);
      if (!clean) continue;
      if (Object.hasOwn(files, clean)) throw new Error(`duplicate archive entry: ${clean}`);
      files[clean] = contents;
    }
    return Folder.fromFiles(files, access);
  }
  const raw = data[0] === 0x1f && data[1] === 0x8b ? gunzipSync(data) : data;
  return Folder.fromFiles(tarFiles(raw), access);
}
