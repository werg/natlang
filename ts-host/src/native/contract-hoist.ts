/**
 * Contract-hoisted structural constraint extraction (`plans/NATLANG_COMPILER.md` §6.4).
 *
 * Inspects a Natlang function's declared return `Type` (and `TypeEnv`) and extracts deterministic structural
 * constraints — exact JSON schema, finite literal unions, required fields, primitive bounds — so they can be
 * enforced on the first generation pass (grammar/schema hint + zero-LLM pre-check before burning repair turns).
 */
import type { Type } from './types.js';
import { TypeEnv } from './types.js';
import { Reject, coerce } from './values.js';

export interface HoistedContract {
  /** Compact JSON-Schema-like object describing the exact structural return shape. */
  jsonSchema: Record<string, unknown>;
  /** Exact JSON-serialized allowed values when the return type is a finite literal union (else null). */
  allowedLiterals: readonly string[] | null;
  /** Compact one-line structural directive for first-pass decoding. */
  directive: string;
  /** Zero-LLM structural check against the hoisted contract (runs before semantic judges / repairs). */
  fastCheck: (value: unknown) => { ok: true } | { ok: false; reason: string };
}

function collectLiterals(type: Type, env: TypeEnv, depth = 0): string[] | null {
  if (depth > 8) return null;
  const resolved = env.resolve(type);
  if (resolved.kind === 'lit') return [JSON.stringify(resolved.value)];
  if (resolved.kind === 'prim' && resolved.name === 'boolean') return ['true', 'false'];
  if (resolved.kind === 'prim' && resolved.name === 'null') return ['null'];
  if (resolved.kind === 'union') {
    const out: string[] = [];
    for (const arm of resolved.members) {
      const sub = collectLiterals(arm, env, depth + 1);
      if (!sub) return null;
      out.push(...sub);
      if (out.length > 64) return null;
    }
    return [...new Set(out)];
  }
  if (resolved.kind === 'refined' || resolved.kind === 'untrusted') {
    return collectLiterals(resolved.base, env, depth + 1);
  }
  return null;
}

function typeToSchema(type: Type, env: TypeEnv, depth = 0): Record<string, unknown> {
  if (depth > 8) return {};
  const resolved = env.resolve(type);
  switch (resolved.kind) {
    case 'prim':
      if (resolved.name === 'string') return { type: 'string' };
      if (resolved.name === 'number') return { type: 'number' };
      if (resolved.name === 'boolean') return { type: 'boolean' };
      if (resolved.name === 'null') return { type: 'null' };
      return {};
    case 'lit':
      return { const: resolved.value };
    case 'list':
      return { type: 'array', items: typeToSchema(resolved.element, env, depth + 1) };
    case 'dict':
      return { type: 'object', additionalProperties: typeToSchema(resolved.element, env, depth + 1) };
    case 'tuple':
      return {
        type: 'array',
        minItems: resolved.minLength ?? resolved.elements.length,
        maxItems: resolved.elements.length,
        prefixItems: resolved.elements.map(el => typeToSchema(el, env, depth + 1)),
      };
    case 'record': {
      const properties: Record<string, unknown> = {};
      const required: string[] = [];
      for (const field of resolved.fields) {
        properties[field.name] = typeToSchema(field.type, env, depth + 1);
        if (!field.optional) required.push(field.name);
      }
      return {
        type: 'object',
        properties,
        ...(required.length ? { required } : {}),
        additionalProperties: false,
      };
    }
    case 'union': {
      const literals = collectLiterals(resolved, env, depth);
      if (literals) return { enum: literals.map(raw => JSON.parse(raw)) };
      return { anyOf: resolved.members.map(arm => typeToSchema(arm, env, depth + 1)) };
    }
    case 'refined':
    case 'untrusted':
      return typeToSchema(resolved.base, env, depth + 1);
    default:
      return {};
  }
}

/** Hoist structural constraints from a function's return type and type environment. */
export function hoistReturnContract(returns: Type, env: TypeEnv): HoistedContract {
  const jsonSchema = typeToSchema(returns, env);
  const allowedLiterals = collectLiterals(returns, env);
  const directive = allowedLiterals
    ? `Structural contract: return exactly one of ${allowedLiterals.join(' | ')}.`
    : jsonSchema.type === 'object' && Array.isArray(jsonSchema.required)
      ? `Structural contract: return an object with required keys [${(jsonSchema.required as string[]).join(', ')}].`
      : `Structural contract: return value matching ${JSON.stringify(jsonSchema)}.`;

  const fastCheck = (value: unknown): { ok: true } | { ok: false; reason: string } => {
    try {
      coerce(value, returns, env, 'return');
      return { ok: true };
    } catch (error) {
      if (error instanceof Reject) return { ok: false, reason: error.message };
      return { ok: false, reason: error instanceof Error ? error.message : String(error) };
    }
  };
  return { jsonSchema, allowedLiterals, directive, fastCheck };
}
