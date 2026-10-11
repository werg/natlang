/** Archive the exact bytes used by a static decision adapter run.
 *
 * Callers declare adapter-specific dependencies. The shared list below covers
 * the reference collector, tool/runtime, materializer, and decision scaffold.
 * The archived payloads are the provenance; the manifest pins both their
 * original resolved paths and their byte-for-byte copies.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../../../');
const host = resolve(here, '../../');

export async function archiveDecisionExecutionComponents({ outDir, adapterPath, additional = [] }) {
  const nativeDir = resolve(host, 'dist/native');
  const nativeNames = (await readdir(nativeDir)).filter(name => name.endsWith('.js')).sort();
  const declared = [
    { path: adapterPath, role: 'adapter-entrypoint' },
    { path: resolve(here, 'execution-components.mjs'), role: 'shared-component-archiver' },
    { path: resolve(here, 'folder-decision-scaffold.mjs'), role: 'decision-folder-scaffold' },
    { path: resolve(here, 'decision-task-contracts.mjs'), role: 'criteria-selection' },
    { path: resolve(repo, 'training/decision_task_contracts.json'), role: 'criteria-contract-data' },
    { path: resolve(here, '../advisory-file.mjs'), role: 'canonical-json-plumbing' },
    { path: resolve(here, '../inline-curriculum/references.mjs'), role: 'reference-collector' },
    { path: resolve(host, 'dist/teacher/collector.js'), role: 'collector-runtime' },
    { path: resolve(host, 'dist/teacher/program.js'), role: 'program-runtime' },
    { path: resolve(host, 'dist/teacher/curriculum.js'), role: 'curriculum-runtime' },
    { path: resolve(host, 'dist/teacher/native-materializer.js'), role: 'native-materializer' },
    { path: resolve(host, 'dist/native/prompt.js'), role: 'native-prompt' },
    { path: resolve(host, 'dist/native/agent.js'), role: 'native-agent' },
    { path: resolve(host, 'dist/scope-compiler.js'), role: 'scope-compiler' },
    { path: resolve(host, 'dist/environment.js'), role: 'runtime-environment' },
    ...nativeNames.map(name => ({ path: resolve(nativeDir, name), role: 'native-tool-or-runtime-module' })),
    ...additional,
  ];
  const byPath = new Map();
  for (const item of declared) {
    const path = resolve(item.path);
    const existing = byPath.get(path);
    if (existing) existing.roles.push(item.role);
    else byPath.set(path, { path, roles: [item.role] });
  }
  const components = [...byPath.values()].sort((a, b) => a.path.localeCompare(b.path));
  const archiveDir = resolve(outDir, 'execution-components');
  const payloadDir = resolve(archiveDir, 'payload');
  await mkdir(payloadDir, { recursive: true });
  const manifestComponents = [];
  for (let index = 0; index < components.length; index++) {
    const component = components[index];
    const bytes = await readFile(component.path); // Fail closed if a declared component is absent.
    const filename = `${String(index + 1).padStart(3, '0')}-${basename(component.path)}`;
    const payloadPath = resolve(payloadDir, filename);
    await writeFile(payloadPath, bytes, { flag: 'wx' });
    manifestComponents.push({ path: component.path, roles: component.roles,
      size_bytes: bytes.length, sha256: sha(bytes), payload: `payload/${filename}`, payload_sha256: sha(await readFile(payloadPath)) });
  }
  const manifest = { schema: 'natlang.static-decision-execution-components/1',
    archive_policy: 'explicit-resolved-paths-only; exact file bytes copied; no directory recursion except the runtime-native modules enumerated by collector tool-surface policy',
    coverage_note: 'Declared execution inputs are pinned; this is not a full transitive dependency closure and does not include the Node runtime, operating system, or package-manager environment.',
    components: manifestComponents };
  const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  const manifestPath = resolve(archiveDir, 'manifest.json');
  await writeFile(manifestPath, manifestBytes, { flag: 'wx' });
  const manifestSha = sha(manifestBytes);
  const pinPath = resolve(archiveDir, 'manifest.sha256');
  await writeFile(pinPath, `${manifestSha}  manifest.json\n`, { flag: 'wx' });
  return { schema: manifest.schema, manifest_path: manifestPath, manifest_sha256: manifestSha,
    component_count: manifestComponents.length, components: manifestComponents };
}

export async function verifyDecisionExecutionComponents(archive) {
  if (archive?.schema !== 'natlang.static-decision-execution-components/1' ||
      !Array.isArray(archive.components) || typeof archive.manifest_path !== 'string' ||
      typeof archive.manifest_sha256 !== 'string') throw new Error('invalid_execution_component_archive_receipt');
  const manifestBytes = await readFile(archive.manifest_path);
  const manifestSha = sha(manifestBytes);
  const manifestPin = (await readFile(resolve(dirname(archive.manifest_path), 'manifest.sha256'), 'utf8')).trim();
  if (manifestSha !== archive.manifest_sha256 || manifestPin !== `${manifestSha}  manifest.json`)
    throw new Error('execution_component_archive_manifest_drift');
  const drift = [];
  for (const component of archive.components) {
    const currentBytes = await readFile(component.path);
    const payloadBytes = await readFile(resolve(dirname(archive.manifest_path), component.payload));
    const currentSha = sha(currentBytes), payloadSha = sha(payloadBytes);
    if (currentSha !== component.sha256 || payloadSha !== component.payload_sha256 ||
        currentSha !== payloadSha || currentBytes.length !== component.size_bytes)
      drift.push({ path: component.path, expected_sha256: component.sha256, current_sha256: currentSha,
        expected_payload_sha256: component.payload_sha256, actual_payload_sha256: payloadSha });
  }
  if (drift.length) {
    const error = new Error(`execution_component_drift_detected:${drift.length}`);
    error.drift = drift;
    throw error;
  }
  return { schema: 'natlang.static-decision-execution-component-verification/1', verified: true,
    checked_components: archive.components.length, manifest_sha256: manifestSha,
    verification: 'current listed source bytes equal the archived payload bytes and their recorded SHA-256 pins' };
}
