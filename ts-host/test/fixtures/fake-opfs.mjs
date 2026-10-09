/**
 * An in-memory stand-in for the origin-private file system (`navigator.storage`): directories, files, `keys()`, and
 * writable streams whose bytes land only when they close (as OPFS's swap files do). `reads` counts `getFile` calls.
 */
export function fakeOpfs() {
  const stats = { reads: 0 };
  const notFound = name => new DOMException(`${name} not found`, 'NotFoundError');
  const makeDirectory = () => {
    const files = new Map(), directories = new Map();
    const directory = {
      files, directories,
      async getDirectoryHandle(name, options = {}) {
        if (!directories.has(name)) {
          if (!options.create) throw notFound(name);
          directories.set(name, makeDirectory());
        }
        return directories.get(name);
      },
      async getFileHandle(name, options = {}) {
        if (!files.has(name)) {
          if (!options.create) throw notFound(name);
          files.set(name, new Uint8Array());
        }
        return {
          async getFile() {
            stats.reads++;
            if (!files.has(name)) throw notFound(name);
            return new Blob([files.get(name)]);
          },
          async createWritable() {
            const chunks = [];
            return {
              async write(chunk) { chunks.push(new Uint8Array(chunk)); },
              async close() { files.set(name, new Uint8Array(Buffer.concat(chunks))); },
              async abort() {},
            };
          },
        };
      },
      async removeEntry(name) {
        if (!files.delete(name) && !directories.delete(name)) throw notFound(name);
      },
      async *keys() { yield* [...files.keys(), ...directories.keys()]; },
    };
    return directory;
  };
  const root = makeDirectory();
  return { root, stats, storage: { async getDirectory() { return root; } } };
}
