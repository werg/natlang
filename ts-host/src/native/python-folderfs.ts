// @ts-nocheck
// An Emscripten filesystem that shows a folder to Pyodide, through a minimal view:
//   kind(path) -> 'file' | 'dir' | null      entries(dir) -> names      size(path) -> bytes
//   read(path) -> Uint8Array                 write(path, bytes)         remove(path)      move(from, to)
//   revision() -> a number that changes whenever the folder changes
// Structure and contents are taken from the view when Python looks them up or opens them; writes go back when a file
// is closed or synced. File nodes are MEMFS nodes, so reading, writing, seeking and mmap reuse MEMFS's code. Cached
// nodes are dropped when the view's revision moves (a change made by another tool, or a child call, while Python was
// waiting on it); an open file keeps its contents, as an open descriptor does.
export function folderFS(py, view, { mtime = Date.UTC(2026, 0, 1) } = {}) {
  const FS = py.FS, MEMFS = FS.filesystems.MEMFS, ERRNO = py._module.ERRNO_CODES;
  const fail = code => { throw new FS.ErrnoError(ERRNO[code]); };
  const emptyDirs = new Set();
  let seen = view.revision(), root;
  const pathOf = node => { const parts = []; for (let n = node; n.parent !== n; n = n.parent) parts.unshift(n.name); return parts.join('/'); };
  const join = (parent, name) => { const base = pathOf(parent); return base ? `${base}/${name}` : name; };
  // Drop every cached node below the root (from Emscripten's name table too), unless the folder is unchanged. Called
  // where other code can have changed the folder: on entry to Python, and when Python resumes after awaiting a call.
  const refresh = () => {
    const now = view.revision();
    if (now === seen) return;
    seen = now;
    const drop = dir => { for (const [name, child] of Object.entries(dir.contents)) {
      if (FS.isDir(child.mode)) drop(child);
      if (!child.dirty) { FS.destroyNode(child); delete dir.contents[name]; }
    } };
    drop(root);
  };
  // Our own writes move the revision too; they keep the cache, which they updated themselves.
  const write = (path, bytes) => { const mine = view.revision() === seen; view.write(path, bytes); if (mine) seen = view.revision(); };

  const fileNode = (parent, name, contents) => {
    const node = MEMFS.createNode(parent, name, 0o100644, 0);
    node.loaded = !!contents; node.contents = contents ?? null; node.usedBytes = contents ? contents.length : -1;
    node.revision = view.revision();
    node.stream_ops = fileStreamOps;
    node.node_ops = { ...MEMFS.ops_table.file.node, getattr, setattr: fileSetattr };
    return node;
  };
  const dirNode = (parent, name, mode = 0o40755) => {
    const node = MEMFS.createNode(parent, name, mode, 0);
    node.node_ops = dirOps;
    return node;
  };
  // Contents are loaded on first open, and again on a later open if the folder changed since and nothing is unsaved.
  const load = node => {
    if (node.loaded && (node.dirty || node.revision === view.revision())) return;
    const bytes = view.read(pathOf(node));
    node.contents = bytes; node.usedBytes = bytes.length; node.loaded = true; node.revision = view.revision();
  };
  const flush = node => {
    if (!node.dirty) return;
    // MEMFS may hold a view of WebAssembly memory (an Int8Array over the heap) that later writes reuse: copy it out.
    const data = node.contents ? node.contents.subarray(0, node.usedBytes) : new Uint8Array();
    write(pathOf(node), new Uint8Array(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)));
    node.dirty = false;
  };
  function getattr(node) {
    if (FS.isFile(node.mode) && !node.loaded && node.usedBytes < 0) node.usedBytes = view.size(pathOf(node));
    const attr = MEMFS.ops_table.file.node.getattr(node);
    attr.atime = attr.mtime = attr.ctime = new Date(mtime);
    return attr;
  }
  function fileSetattr(node, attr) {
    if (attr.size !== undefined) { load(node); node.dirty = true; }
    return MEMFS.ops_table.file.node.setattr(node, attr);
  }
  const memStream = MEMFS.ops_table.file.stream;
  const fileStreamOps = {
    ...memStream,
    open(stream) { load(stream.node); },
    write(stream, ...rest) { stream.node.dirty = true; return memStream.write(stream, ...rest); },
    msync(stream, ...rest) { const out = memStream.msync(stream, ...rest); flush(stream.node); return out; },
    close(stream) { flush(stream.node); },
  };
  const dirOps = {
    ...MEMFS.ops_table.dir.node,
    getattr,
    lookup(parent, name) {
      if (parent.contents[name]) return parent.contents[name];
      const path = join(parent, name), kind = view.kind(path);
      if (kind === 'file') return fileNode(parent, name);
      if (kind === 'dir' || emptyDirs.has(path)) return dirNode(parent, name);
      fail('ENOENT');
    },
    readdir(node) {
      const base = pathOf(node), prefix = base ? `${base}/` : '';
      const names = new Set([...Object.keys(node.contents), ...view.entries(base)]);
      for (const dir of emptyDirs) if (dir.startsWith(prefix) && !dir.slice(prefix.length).includes('/')) names.add(dir.slice(prefix.length));
      return ['.', '..', ...[...names].sort()];
    },
    mknod(parent, name, mode) {
      const path = join(parent, name);
      if (FS.isDir(mode)) { emptyDirs.add(path); return dirNode(parent, name, mode); }
      write(path, new Uint8Array());
      return fileNode(parent, name, new Uint8Array());
    },
    rename(oldNode, newDir, newName) {
      const from = pathOf(oldNode), to = join(newDir, newName);
      if (FS.isFile(oldNode.mode)) flush(oldNode);
      if (view.kind(to) === 'file') view.remove(to);
      if (FS.isDir(oldNode.mode) && emptyDirs.has(from) && view.kind(from) !== 'dir') { emptyDirs.delete(from); emptyDirs.add(to); }
      else view.move(from, to);
      seen = view.revision();
      delete oldNode.parent.contents[oldNode.name];
      oldNode.name = newName; oldNode.parent = newDir; newDir.contents[newName] = oldNode;
    },
    unlink(parent, name) { view.remove(join(parent, name)); seen = view.revision(); delete parent.contents[name]; },
    rmdir(parent, name) {
      const path = join(parent, name);
      if (view.entries(path).length) fail('ENOTEMPTY');
      emptyDirs.delete(path); delete parent.contents[name];
    },
    symlink() { fail('EPERM'); },
  };
  return { mount() { root = dirNode(null, '/'); return root; }, refresh };
}

/** The folder's public metadata and byte operations, shared with shell and file tools. */
export function folderView(folder) {
  return {
    revision: () => folder.revision(),
    kind: path => folder.kind(path),
    entries: dir => folder.childNames(dir),
    size: path => folder.size(path),
    read: path => folder.readBytesSync(path),
    write: (path, bytes) => folder.writeBytes(path, bytes),
    remove: path => folder.remove(path),
    move: (from, to) => folder.move(from, to),
  };
}
