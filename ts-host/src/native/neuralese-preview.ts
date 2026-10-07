/** Resolve a text-emulation preview only back to a Neuralese reference already visible in this call. */
import { hexDigest } from './hash.js';
import { isNeuraleseRef, type NeuraleseRef } from './neuralese.js';
import type { NeuraleseStore } from './neuralese-store.js';
import { fitsType, parseType, TypeEnv, type Type } from './types.js';

export type NeuralesePreviewResolution = { path: string; id: string; type: string; body_sha256: string };
export type NeuralesePreviewResult = { value: unknown; resolutions: NeuralesePreviewResolution[] };

type Label = { id: string; type: string; body: string };

function parseLabel(value: string): Label | undefined {
  const match = /^\[\[Neuralese text block id=(nz1_[a-z2-7]{20,}) type=([\s\S]+); exact JSON string body=("(?:[^"\\]|\\.)*")\]\]$/.exec(value);
  if (!match) return;
  try {
    const body: unknown = JSON.parse(match[3]!);
    if (typeof body !== 'string' || JSON.stringify(body) !== match[3]) return;
    return { id: match[1]!, type: match[2]!, body };
  } catch { return; }
}

const plainRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' &&
  !Array.isArray(value) && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

/**
 * Replace only complete text-emulation preview labels that identify an exact typed reference already in this call's
 * arguments, locals, or captures. Store presence alone is never authority. The caller still applies normal `coerce`.
 */
export async function resolveNeuralesePreviews(value: unknown, expected: Type, env: TypeEnv,
  visibleValues: readonly unknown[], store: NeuraleseStore): Promise<NeuralesePreviewResult> {
  const visible = new Map<string, NeuraleseRef[]>();
  const scanned = new Set<object>();
  const collect = (item: unknown): void => {
    if (isNeuraleseRef(item)) {
      const refs = visible.get(item.$neuralese.id) ?? [];
      if (!refs.some(ref => ref.$neuralese.type === item.$neuralese.type)) refs.push(item);
      visible.set(item.$neuralese.id, refs);
      return;
    }
    if (!item || typeof item !== 'object' || scanned.has(item)) return;
    scanned.add(item);
    if (Array.isArray(item)) item.forEach(collect);
    else if (plainRecord(item)) Object.values(item).forEach(collect);
  };
  visibleValues.forEach(collect);

  const resolutions: NeuralesePreviewResolution[] = [];
  const atType = async (item: unknown, type: Type, path: string): Promise<unknown> => {
    let resolved: Type;
    try { resolved = env.resolve(type); } catch { return item; }
    if (typeof item === 'string' && resolved.kind === 'neuralese') {
      const label = parseLabel(item);
      if (!label) return item;
      const refs = visible.get(label.id) ?? [];
      const matches: { ref: NeuraleseRef; body_sha256: string }[] = [];
      for (const ref of refs) {
        if (ref.$neuralese.type !== label.type) continue;
        let sourceType: Type;
        try { sourceType = parseType(label.type); } catch { continue; }
        if (sourceType.kind !== 'neuralese' || !fitsType(sourceType, resolved, env)) continue;
        const meta = await store.meta(label.id);
        const producer = meta?.producer;
        const bodySha = hexDigest(label.body);
        if (!meta || meta.id !== label.id || meta.type !== label.type || producer?.kind !== 'text-marker-emulation' ||
            producer.emulation_version !== 'text-marker-standin/2' || producer.text_body_sha256 !== bodySha) continue;
        matches.push({ ref, body_sha256: bodySha });
      }
      const distinct = new Map(matches.map(match => [match.ref.$neuralese.type, match]));
      if (distinct.size !== 1) return item;
      const found = [...distinct.values()][0]!;
      resolutions.push({ path, id: label.id, type: label.type, body_sha256: found.body_sha256 });
      return found.ref;
    }
    if (resolved.kind === 'list' && Array.isArray(item))
      return Promise.all(item.map((child, index) => atType(child, resolved.element, `${path}/${index}`)));
    if (resolved.kind === 'dict' && plainRecord(item)) {
      const out: Record<string, unknown> = Object.create(Object.getPrototypeOf(item));
      for (const [key, child] of Object.entries(item)) Object.defineProperty(out, key, { enumerable: true, configurable: true,
        writable: true, value: await atType(child, resolved.element, `${path}/${key}`) });
      return out;
    }
    if (resolved.kind === 'record' && plainRecord(item)) {
      const fields = new Map(resolved.fields.map(field => [field.name, field.type]));
      const out: Record<string, unknown> = Object.create(Object.getPrototypeOf(item));
      for (const [key, child] of Object.entries(item)) Object.defineProperty(out, key, { enumerable: true, configurable: true,
        writable: true, value: fields.has(key) ? await atType(child, fields.get(key)!, `${path}/${key}`) : child });
      return out;
    }
    return item;
  };

  const normalized = await atType(value, expected, 'return');
  // An unchanged result needs no copy or special handling.
  return { value: resolutions.length ? normalized : value, resolutions };
}
