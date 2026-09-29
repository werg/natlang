import type { AdaptationBinding, ComponentValue, ProgramDescriptor } from './types.js';
import type { ItemRecord, NatlangRecord } from '../runtime/loader.js';
import { fingerprint } from './identity.js';
import type { InlineLambdaPlan } from '../compiler/inline.js';
import { inlineDescriptor } from './inventory.js';
/** Frontmatter is copied byte for byte. */
export function replaceNamedBody(source: string, body: string): string {
  const match = /^(---\r?\n[\s\S]*?\r?\n---\r?\n?)/.exec(source);
  if (!match) throw new Error('cannot project a named instruction without frontmatter');
  return match[1] + body;
}
/** Escapes only static segments; original interpolation expression AST text is retained. */
export function templateSource(segments: readonly string[], expressions: readonly string[]): string {
  const escape = (text: string) => text.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');
  return '`' + segments.map((segment, index) => escape(segment) + (index < expressions.length ? '${' + expressions[index] + '}' : '')).join('') + '`';
}
export class ProgramView {
  private readonly revisions = new Map<string, ItemRecord>();
  private revision = 0;
  private cache = new WeakMap<ItemRecord, ItemRecord>();
  private originals = new WeakMap<ItemRecord, ItemRecord>();
  private inlineRevisions = new Map<string, InlineLambdaPlan | null>();
  component(definitionId: string) {
    return this.program?.components.find(item => item.definitionId === definitionId ||
      !!item.definitionId && this.inlineRevisions.get(item.definitionId)?.definitionId === definitionId);
  }
  inlineRevision(plan: InlineLambdaPlan): InlineLambdaPlan {
    if (!this.inlineRevisions.has(plan.definitionId)) return plan;
    const current = this.inlineRevisions.get(plan.definitionId);
    if (!current || !plan.adaptation || !current.adaptation ||
      inlineDescriptor('', plan).contractHash !== inlineDescriptor('', current).contractHash)
      throw new Error('runtime source edit changed an existing inline callable contract; recreate the callable');
    return current;
  }
  original<T extends ItemRecord>(record: T): T { return (this.originals.get(record) ?? record) as T; }
  constructor(readonly program?: ProgramDescriptor, readonly binding?: AdaptationBinding | null) {}
  value(definitionId: string, owner?: string): ComponentValue | undefined {
    if (owner && owner !== this.program?.id) return undefined;
    const component = this.component(definitionId);
    if (component?.source && this.revisions.has(component.source.path)) return undefined;
    return component ? this.binding?.candidate[component.key] : undefined;
  }
  guidance(): string {
    const component = this.program?.components.find(item => item.kind === 'program.guidance');
    const value = component && (this.binding?.candidate[component.key] ?? component.baseline);
    return value?.kind === 'program.guidance' ? value.text : '';
  }
  source(path: string, fallback: string): string {
    const revision = this.revisions.get(path); if (revision && revision.kind !== 'namespace') return revision.text;
    let source = this.program?.sources[path] ?? fallback;
    const edits: { start: number; end: number; text: string }[] = [];
    for (const descriptor of this.program?.components ?? []) {
      if (descriptor.source?.path !== path) continue;
      const value = this.binding?.candidate[descriptor.key]; if (value?.kind !== 'lambda.instructions') continue;
      if (fingerprint(value) === descriptor.baselineHash) continue;
      if (descriptor.origin === 'named') return replaceNamedBody(source, value.template.segments[0]!);
      const location = descriptor.source;
      if (location.templateStart === undefined || location.templateEnd === undefined) throw new Error('missing template origin mapping');
      edits.push({ start: location.templateStart, end: location.templateEnd, text: templateSource(value.template.segments, location.expressions ?? []) });
    }
    for (const edit of edits.sort((a, b) => b.start - a.start)) source = source.slice(0, edit.start) + edit.text + source.slice(edit.end);
    return source;
  }
  record<T extends ItemRecord>(record: T): T {
    if (record.kind !== 'namespace' && record.programId && record.programId !== this.program?.id) return record;
    const revision = this.revisions.get(record.source); if (revision) return revision as T;
    if (!this.binding && !this.revision) return record;
    if (this.originals.has(record)) return record;
    const cached = this.cache.get(record); if (cached) return cached as T;
    const expected = this.program?.sources[record.source];
    if (this.binding && record.kind !== 'namespace' && expected !== undefined && record.text !== expected)
      throw new Error('loaded callable source differs from bound program: ' + record.source);
    const children = this.tree(record.codebase);
    if (record.kind === 'namespace') return { ...record, codebase: children };
    const text = this.source(record.source, record.text);
    if (record.kind === 'natlang') {
      const value = this.value(record.id, record.programId);
      const effective = { ...record, text, codebase: children, ...(value?.kind === 'lambda.instructions' ? { instructions: value.template.segments[0]! } : {}) };
      this.cache.set(record, effective); this.originals.set(effective, record); return effective;
    }
    const effective = { ...record, text, codebase: children };
    this.cache.set(record, effective); this.originals.set(effective, record); return effective;
  }
  tree(records: Record<string, ItemRecord>): Record<string, ItemRecord> {
    return Object.fromEntries(Object.entries(records).map(([name, record]) => [name, this.record(record)]));
  }
  patched(path: string): boolean { return this.revisions.has(path); }
  get revisionId(): number { return this.revision; }
  commit(record: ItemRecord, expectedRevision: number,
    inline?: { original: InlineLambdaPlan[]; previous: InlineLambdaPlan[]; current: InlineLambdaPlan[] }): number {
    if (expectedRevision !== this.revision) throw new Error('runtime source edit conflict; reread current source');
    if (inline) {
      const match = (plan: InlineLambdaPlan, index: number, count: number) => plan.adaptation?.label ?
        inline.current.find(item => item.adaptation?.label === plan.adaptation!.label) ?? null :
        count === inline.current.length ? inline.current[index] ?? null : null;
      for (const [id, previous] of this.inlineRevisions) {
        if (previous?.sourceSpan.file === record.source)
          this.inlineRevisions.set(id, match(previous, inline.previous.findIndex(item => item.definitionId === previous.definitionId), inline.previous.length));
      }
      for (const plans of [inline.original, inline.previous]) plans.forEach((plan, index) =>
        this.inlineRevisions.set(plan.definitionId, match(plan, index, plans.length)));
    }
    this.revisions.set(record.source, record); this.cache = new WeakMap(); return ++this.revision;
  }
  provenance(definitionId: string, owner?: string): Record<string, unknown> {
    const component = (!owner || owner === this.program?.id) ? this.component(definitionId) : undefined;
    const path = component?.source?.path;
    const original = path ? this.program?.sources[path] : undefined;
    const patched = path ? this.revisions.get(path) : undefined;
    return { program: this.program?.id ?? null, component: component?.key ?? null, buildHash: this.program?.buildHash ?? null,
      artifact: this.binding?.artifact.digest ?? null, instructionHash: fingerprint(this.value(definitionId, owner) ?? component?.baseline ?? null),
      guidanceHash: fingerprint(this.guidance()), runtimeRevision: this.revision, executor: this.binding?.executor ?? null,
      originalSourceHash: original === undefined ? null : fingerprint(original),
      effectiveSourceHash: original === undefined || !path ? null : fingerprint(this.source(path, original)),
      patchedSourceHash: patched && patched.kind !== 'namespace' ? fingerprint(patched.text) : null };
  }
}
