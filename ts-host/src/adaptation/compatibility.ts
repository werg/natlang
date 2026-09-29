import type { AdaptationBinding, Candidate, ComponentDescriptor, ExecutorIdentity, ProgramDescriptor } from './types.js';
import { fingerprint, canonical, cloneData, immutable } from './identity.js';
import { AdaptationError, validateAdaptation, validateValue } from './schema.js';
const fail = (text: string): never => { throw new AdaptationError(text); };
const mentions = (text: string, name: string) => new RegExp('(^|[^A-Za-z0-9_$])' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?![A-Za-z0-9_$])').test(text);
export function validateCandidate(candidate: Candidate, descriptors: readonly ComponentDescriptor[], complete = true): Candidate {
  const known = new Map(descriptors.map(component => [component.key, component]));
  if (complete && known.size !== Object.keys(candidate).length) fail('candidate must contain the exact selected component set');
  for (const [key, raw] of Object.entries(candidate)) {
    const descriptor = known.get(key); if (!descriptor) fail('unknown component: ' + key);
    const d = descriptor!; const value = validateValue(raw); if (value.kind !== d.kind) fail('wrong component kind: ' + key);
    const text = value.kind === 'program.guidance' ? value.text : value.template.segments.join('');
    if (text.length > d.constraints.maxChars) fail('instruction size limit exceeded: ' + key);
    if (value.kind === 'lambda.instructions') {
      if (canonical(value.template.slotIds) !== canonical(d.contract.slots)) fail('slot identity changed: ' + key);
      if (fingerprint(value) === d.baselineHash) continue;
      const allowedNames = new Set([...d.contract.visibleBindings, ...d.contract.parameters.map(parameter => parameter.name), 'result', 'nl', 'iterateOn', 'self']);
      for (const segment of value.template.segments) for (const match of segment.matchAll(/`([A-Za-z_$][A-Za-z0-9_$]*)`/g))
        if (!allowedNames.has(match[1]!)) fail('unknown quoted lexical binding: ' + match[1]);
      const captureNames = new Set(d.contract.captures.map(capture => capture.name));
      for (const name of d.contract.visibleBindings) {
        const present = value.template.segments.some(segment => mentions(segment, name)) || d.contract.slotBindings.includes(name);
        if (present !== captureNames.has(name)) fail('capture contract changed for ' + name + ': ' + key);
      }
    }
    for (const name of d.constraints.requiredBindings) if (!mentions(text, name)) fail('required binding missing: ' + name);
  }
  return immutable(cloneData(candidate));
}
const bindings = new WeakSet<object>();
export function isAdaptationBinding(value: unknown): value is AdaptationBinding { return !!value && typeof value === 'object' && bindings.has(value); }
export function bindAdaptation(input: unknown, program: ProgramDescriptor, executor: ExecutorIdentity): AdaptationBinding {
  const artifact = validateAdaptation(input);
  if (program.schema !== 'natlang.program/v1' || fingerprint(program.sources, 'natlang.build/v1') !== program.buildHash)
    fail('invalid program build metadata');
  if (new Set(program.components.map(component => component.key)).size !== program.components.length) fail('duplicate program component key');
  for (const component of program.components) if (fingerprint(component.baseline) !== component.baselineHash || fingerprint(component.contract) !== component.contractHash)
    fail('invalid component metadata: ' + component.key);
  if (artifact.program.id !== program.id || artifact.program.buildHash !== program.buildHash || artifact.program.protocol !== program.protocol)
    fail('adaptation is stale for this program; revalidate against the current build');
  if (canonical(artifact.program.guidanceScope) !== canonical(program.guidanceScope)) fail('guidance scope mismatch');
  if (fingerprint(artifact.executor) !== fingerprint(executor)) fail('executor identity/configuration mismatch; revalidation required');
  const known = new Map(program.components.map(component => [component.key, component]));
  const candidate: Record<string, import('./types.js').ComponentValue> = Object.create(null);
  for (const entry of artifact.components) {
    const descriptor = known.get(entry.key); if (!descriptor) fail('unknown component: ' + entry.key);
    if (descriptor!.baselineHash !== entry.baselineHash || descriptor!.contractHash !== entry.contractHash) fail('component contract/baseline mismatch: ' + entry.key);
    candidate[entry.key] = entry.value;
  }
  const binding = immutable({ artifact, program: immutable(cloneData(program)), executor: immutable(cloneData(executor)),
    candidate: validateCandidate(candidate, program.components, false) }); bindings.add(binding); return binding;
}
export function inspectComponents(program: ProgramDescriptor): readonly ComponentDescriptor[] { return program.components; }
