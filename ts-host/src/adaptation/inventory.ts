import type { InlineLambdaPlan } from '../compiler/inline.js';
import type { ItemRecord, NatlangRecord } from '../runtime/loader.js';
import { NATLANG_COMPILE_VERSION } from '../compiler/intrinsics.js';
import type { ComponentDescriptor, ComponentValue, FrozenComponentContract, ProgramDescriptor } from './types.js';
import { componentKey, fingerprint, immutable, canonical } from './identity.js';
const baseContract = (): FrozenComponentContract => ({ parameters: [], returns: 'null', types: {}, subtype: 'function',
  openParameters: false, captures: [], slots: [], helpers: [], servicesHash: fingerprint({}), protocol: NATLANG_COMPILE_VERSION,
  visibleBindings: [], slotBindings: [] });
function descriptor(key: string, origin: ComponentDescriptor['origin'], baseline: ComponentValue,
  contract: FrozenComponentContract, other: Partial<ComponentDescriptor> = {}): ComponentDescriptor {
  return { key, kind: baseline.kind, origin, baseline, baselineHash: fingerprint(baseline), contract, contractHash: fingerprint(contract),
    constraints: { maxChars: 32000, requiredBindings: [] }, ...other };
}
export function namedDescriptor(program: string, record: NatlangRecord): ComponentDescriptor {
  const helpers = (records: Record<string, ItemRecord>): string[] => Object.values(records).flatMap(item =>
    [item.source, ...helpers(item.codebase)]).sort();
  return descriptor(componentKey(program, record.source), 'named',
    { kind: 'lambda.instructions', template: { segments: [record.instructions], slotIds: [] } },
    { ...baseContract(), parameters: Object.entries(record.args).map(([name, type]) => ({ name, type })),
      returns: record.returns, types: record.types, subtype: record.subtype, helpers: helpers(record.codebase) },
    { definitionId: record.id, source: { path: record.source, start: 0, end: record.text.length } });
}
export function inlineDescriptor(program: string, plan: InlineLambdaPlan): ComponentDescriptor {
  if (!plan.adaptation) throw new Error('runtime-generated inline sites cannot be selected persistently');
  const { expressions, visibleBindings, slotBindings, ...location } = plan.adaptation;
  const slots = expressions.map(text => fingerprint(text, 'natlang.slot/v1'));
  const site = plan.adaptation.label ? 'site:' + plan.adaptation.label : 'exact:' + plan.definitionId;
  return descriptor(componentKey(program, plan.sourceSpan.file, site + ':instructions'), 'authored-inline',
    { kind: 'lambda.instructions', template: { segments: plan.strings, slotIds: slots } },
    { ...baseContract(), parameters: plan.parameters, returns: plan.returns, openParameters: !!plan.openParameters,
      captures: plan.captures.map(({ name, type, mutable }) => ({ name, type, mutable })).sort((a, b) => a.name.localeCompare(b.name)), slots, visibleBindings, slotBindings },
    { definitionId: plan.definitionId, source: { path: plan.sourceSpan.file, start: plan.sourceSpan.start,
      end: plan.sourceSpan.end, templateStart: location.templateStart, templateEnd: location.templateEnd, expressions } });
}
export function describeProgram(id: string, sources: Record<string, string>, components: ComponentDescriptor[],
  guidance = '', importedPrograms: readonly string[] = [], services?: ProgramDescriptor['services']): ProgramDescriptor {
  if (guidance || importedPrograms.length || services) sources = { ...sources,
    '@natlang/program-policy.json': canonical({ guidance, importedPrograms, services: services ?? null }) };
  const guidanceComponent = descriptor(encodeURIComponent(id) + '::program.guidance', 'program', { kind: 'program.guidance', text: guidance }, baseContract());
  // Source hash excludes outputs, artifact pointers and generated declarations at the compiler boundary.
  const program: ProgramDescriptor = { schema: 'natlang.program/v1', id, protocol: NATLANG_COMPILE_VERSION,
    buildHash: fingerprint(sources, 'natlang.build/v1'), sources, components: [...components, guidanceComponent].sort((a, b) => a.key.localeCompare(b.key)),
    guidanceScope: { importedPrograms }, ...(services ? { services } : {}) };
  if (services) {
    const servicesHash = fingerprint(services, 'natlang.services/v1');
    program.components = program.components.map(component => {
      const contract = { ...component.contract, servicesHash };
      return { ...component, contract, contractHash: fingerprint(contract) };
    });
  }
  if (new Set(program.components.map(component => component.key)).size !== program.components.length) throw new Error('duplicate optimization identity');
  return immutable(program);
}
