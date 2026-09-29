import { resolve, relative, isAbsolute, sep } from 'node:path';
export { loadAdaptation } from '../adaptation/node.js';
import { validateAdaptation, artifactDigest } from '../adaptation/schema.js';
import { bindAdaptation, validateCandidate } from '../adaptation/compatibility.js';
import { ProgramView } from '../adaptation/program-view.js';
import type { AdaptationArtifact, ProgramDescriptor, ExecutorIdentity, Candidate } from '../adaptation/types.js';
import { fingerprint } from '../adaptation/identity.js';
import { buildProject, nodeProjectFiles } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { evaluate, candidateArtifact } from '../evaluation/runner.js';
import type { PreparedSuite } from '../evaluation/types.js';
import { pairedChanges } from '../evaluation/report.js';
import type { ModelDriver } from '../runtime/runtime.js';
import { UsageGateway } from '../evaluation/usage.js';
import { selectionEligible } from './strategies/gepa.js';
export function exportAdaptationPatch(artifact: AdaptationArtifact, program: ProgramDescriptor, executor: ExecutorIdentity,
  root: string): { patch: string; manifest: readonly { path: string; original: string; updated: string }[]; guidance?: string } {
  root = resolve(root);
  const binding = bindAdaptation(artifact, program, executor); const view = new ProgramView(program, binding);
  for (const [path, original] of Object.entries(program.sources)) {
    if (path.startsWith('@')) continue;
    const file = resolve(root, path);
    const location = relative(root, file);
    if (location === '..' || location.startsWith('..' + sep) || isAbsolute(location)) throw new Error('program source escapes export root: ' + path);
    if (!nodeProjectFiles.isFile(file)) throw new Error('program source is missing; rebuild/revalidate: ' + path);
    let actual = nodeProjectFiles.read(file);
    if (path.split('/').at(-1) === 'natlang.json') {
      try { const manifest = JSON.parse(actual); if (manifest.targets) for (const target of Object.values(manifest.targets) as Record<string, unknown>[]) delete target.adaptation;
        actual = JSON.stringify(manifest); } catch { /* Invalid manifest changes remain stale source. */ }
    }
    if (actual !== original) throw new Error('program source changed; rebuild/revalidate before source export: ' + path);
  }
  const changed: Record<string, string> = {}; const manifest: { path: string; original: string; updated: string }[] = [];
  let patch = '';
  for (const [path, original] of Object.entries(program.sources)) {
    const updated = view.source(path, original); if (updated === original) continue; changed[path] = updated;
    manifest.push({ path, original: fingerprint(original), updated: fingerprint(updated) });
    const lines = (text: string) => text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
    const before = lines(original), after = lines(updated);
    const emit = (line: string, prefix: string) => prefix + line + (line.endsWith('\n') ? '' : '\n\\ No newline at end of file\n');
    patch += '--- a/' + path + '\n+++ b/' + path + '\n@@ -1,' + before.length + ' +1,' + after.length + ' @@\n' +
      before.map(line => emit(line, '-')).join('') + after.map(line => emit(line, '+')).join('');
  }
  const result = buildProject({ project: root, programId: program.id, services: program.services, emit: false, write: false, files: { ...nodeProjectFiles,
    read: path => { const relative = path.slice(root.replace(/\/$/, '').length + 1); return changed[relative] ?? nodeProjectFiles.read(path); } } });
  if (!result.ok) throw new Error('exported source does not compile: ' + formatDiagnostics(result.diagnostics));
  for (const path of new Set(program.components.map(component => component.source?.path).filter(Boolean))) {
    const before = program.components.filter(component => component.source?.path === path).sort((a, b) => a.source!.start - b.source!.start);
    const after = result.manifest.adaptation!.components.filter(component => component.source?.path === path).sort((a, b) => a.source!.start - b.source!.start);
    if (before.length !== after.length || before.some((component, index) => component.contractHash !== after[index]?.contractHash))
      throw new Error('source export changed frozen contracts: ' + path);
  }
  const guidance = view.guidance();
  const selectedGuidance = artifact.components.some(component => component.value.kind === 'program.guidance');
  return { patch, manifest, ...(guidance || selectedGuidance ? { guidance } : {}) };
}
/** Explicit contract mapping plus new regression evidence; executor/source changes issue a new artifact. */
export async function revalidateAdaptation(old: AdaptationArtifact, prepared: PreparedSuite, driver: ModelDriver,
  mapping: Readonly<Record<string, string>> = {}, options: { judge?: ModelDriver; signal?: AbortSignal } = {}): Promise<AdaptationArtifact> {
  old = validateAdaptation(old);
  const candidate: Record<string, Candidate[string]> = {};
  const selected = prepared.program.components.filter(component => prepared.components.includes(component.key));
  const mapped = new Set<string>();
  for (const component of selected) candidate[component.key] = component.baseline;
  for (const entry of old.components) {
    const key = mapping[entry.key] ?? entry.key; const component = selected.find(component => component.key === key);
    if (mapped.has(key)) throw new Error('revalidation component mappings must be one-to-one: ' + key);
    mapped.add(key);
    if (!component || component.contractHash !== entry.contractHash) throw new Error('revalidation needs compatible explicit component mapping: ' + entry.key);
    candidate[key] = entry.value;
  }
  validateCandidate(candidate, selected);
  exportAdaptationPatch(candidateArtifact(prepared, candidate), prepared.program, prepared.suite.executorIdentity, prepared.suite.program.root);
  const gateway = new UsageGateway(prepared.suite.budget);
  const baseline = await evaluate(prepared, { split: 'validation', driver, gateway, ...options });
  const batch = await evaluate(prepared, { split: 'validation', candidate, driver, gateway, ...options });
  const changedGuidance = selected.some(component => component.kind === 'program.guidance' &&
    fingerprint(candidate[component.key]) !== component.baselineHash);
  const coverage = new Set(batch.results.flatMap(result => [...result.coverage]));
  if (changedGuidance && prepared.program.components.some(component => component.kind === 'lambda.instructions' && !coverage.has(component.key)))
    throw new Error('revalidation failed whole-program guidance coverage; exercise every authored lambda');
  if (!selectionEligible({ id: 'revalidation', value: candidate, parents: [], train: batch, validation: batch }, prepared.suite.selection) ||
    batch.quality === null || baseline.quality !== null && batch.quality < baseline.quality)
    throw new Error('revalidation failed required regression gates or paired baseline quality');
  const artifact = candidateArtifact(prepared, candidate, 'revalidate-' + old.digest.slice(0, 12));
  artifact.provenance = { ...artifact.provenance, promotion: 'revalidated', evidence: { previous: old.digest,
    baseline: baseline.digest, validation: batch.digest, quality: batch.quality, ledger: gateway.snapshot(), pairedChanges: pairedChanges(baseline, batch) } };
  artifact.digest = artifactDigest(artifact); return artifact;
}
