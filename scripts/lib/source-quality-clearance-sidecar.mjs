import { readFile } from 'node:fs/promises';
import { isAbsolute, resolve, relative } from 'node:path';
import { sourceQualityClearanceRefProblems, verifySourceQualityClearanceSidecar } from '../../ts-host/dist/teacher/source-quality-clearance.js';

function packetPath(root, value) {
  if (typeof value !== 'string' || !value || isAbsolute(value) || value.includes('\\'))
    throw new Error('source-quality-clearance-sidecar-path-invalid');
  const path = resolve(root, value), rel = relative(root, path);
  if (!rel || rel === '..' || rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
    throw new Error('source-quality-clearance-sidecar-path-escapes-packet');
  return path;
}

/** Resolve a row's compact reference through its packet manifest and verify the signed sidecar plus evidence bytes. */
export async function resolvePacketSourceQualityClearance(packetRootValue, reference, program) {
  const root = resolve(packetRootValue);
  const manifestPath = packetPath(root, 'materialized-manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const sidecar = manifest.source_quality_clearance;
  if (!sidecar || sidecar.path !== reference?.sidecar_path || sidecar.file_sha256 !== reference?.file_sha256 ||
      sidecar.review_canonical_sha256 !== reference?.review_canonical_sha256 || !Array.isArray(sidecar.evidence))
    throw new Error('source-quality-clearance-packet-manifest-mismatch');
  const bytes = await readFile(packetPath(root, sidecar.path));
  const evidence = new Map();
  for (const entry of sidecar.evidence) {
    if (!entry || typeof entry.sha256 !== 'string' || evidence.has(entry.sha256))
      throw new Error('source-quality-clearance-packet-evidence-invalid');
    evidence.set(entry.sha256, await readFile(packetPath(root, entry.path)));
  }
  const clearance = verifySourceQualityClearanceSidecar(bytes, sidecar.file_sha256, evidence);
  if (clearance.review_canonical_sha256 !== sidecar.review_canonical_sha256)
    throw new Error('source-quality-clearance-packet-canonical-hash-mismatch');
  const problems = sourceQualityClearanceRefProblems(reference, clearance, program);
  if (problems.length) throw new Error(problems.join(','));
  return clearance;
}
