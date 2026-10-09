import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

/**
 * Verify a schema-neutral closure manifest and require the requested artifact
 * to be one of its pinned files. Paths are resolved relative to the manifest;
 * every listed byte hash (and optional byte count) is checked.
 */
export function verifyPinnedCodeClosure(manifestPath, artifactPath) {
  const absoluteManifest = resolve(manifestPath);
  const manifestBytes = readFileSync(absoluteManifest);
  const manifest = JSON.parse(manifestBytes);
  const entries = manifest?.closure_files;
  if (!Array.isArray(entries) || entries.length === 0)
    throw new Error('code manifest must contain a nonempty closure_files array');

  const manifestDir = dirname(absoluteManifest);
  const seen = new Set();
  let artifactEntry = null;
  const verified = [];
  for (const entry of entries) {
    if (!entry || typeof entry.path !== 'string' || !entry.path || isAbsolute(entry.path) ||
        typeof entry.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(entry.sha256))
      throw new Error('invalid code-closure entry');
    const path = resolve(manifestDir, entry.path);
    const fromManifest = relative(manifestDir, path);
    if (!fromManifest || fromManifest === '..' || fromManifest.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
      throw new Error(`code-closure path escapes manifest directory: ${entry.path}`);
    if (seen.has(path)) throw new Error(`duplicate code-closure path: ${entry.path}`);
    seen.add(path);
    const contents = readFileSync(path);
    const actual = sha256(contents);
    if (actual !== entry.sha256) throw new Error(`code-closure hash mismatch: ${entry.path}`);
    if (entry.bytes !== undefined && entry.bytes !== contents.length)
      throw new Error(`code-closure byte count mismatch: ${entry.path}`);
    const record = { path: entry.path, sha256: actual, bytes: contents.length };
    verified.push(record);
    if (path === resolve(artifactPath)) artifactEntry = record;
  }
  if (!artifactEntry) throw new Error('requested artifact is not pinned by the code manifest');
  return {
    manifest_path: absoluteManifest,
    manifest_sha256: sha256(manifestBytes),
    artifact: artifactEntry,
    closure_files_verified: verified.length,
    closure_sha256: sha256(Buffer.from(JSON.stringify(verified)))
  };
}
