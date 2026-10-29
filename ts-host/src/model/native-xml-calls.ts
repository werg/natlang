/** Exact complete LFM3/Qwen XML calls, using the same first-close-delimiter law as serve/chat.py. */
type Schema = Record<string, unknown>;
type Tool = { function: { name: string; parameters?: unknown } };
const OPEN = '<tool_call>';
const CLOSE = '\n</parameter>\n';
const annotation = new Set(['description', 'title', 'default', 'examples', '$schema', 'deprecated', 'readOnly', 'writeOnly']);

function types(schema: Schema): string[] | undefined {
  if (typeof schema.type === 'string') return [schema.type];
  if (Array.isArray(schema.type) && schema.type.every(x => typeof x === 'string')) return schema.type as string[];
  for (const name of ['anyOf', 'oneOf']) if (Array.isArray(schema[name])) {
    const parts = (schema[name] as Schema[]).map(types);
    return parts.some(x => x === undefined) ? undefined : [...new Set(parts.flat() as string[])];
  }
  return undefined;
}
function matches(type: string, value: unknown): boolean {
  switch (type) {
    case 'null': return value === null;
    case 'object': return !!value && typeof value === 'object' && !Array.isArray(value);
    case 'array': return Array.isArray(value);
    case 'string': return typeof value === 'string';
    case 'boolean': return typeof value === 'boolean';
    case 'integer': return typeof value === 'number' && Number.isSafeInteger(value);
    case 'number': return typeof value === 'number' && Number.isFinite(value);
    default: return false;
  }
}
function valueFor(text: string, schema: Schema): unknown {
  const kinds = types(schema);
  if (kinds?.length === 1 && kinds[0] === 'string') return text;
  let value: unknown;
  try { value = JSON.parse(text); }
  catch { if (kinds && !kinds.includes('string')) throw new Error('parameter is not declared JSON type'); return text; }
  if (!kinds) return typeof value === 'string' ? (JSON.stringify(value) === text ? value : text) : value;
  if (kinds.some(kind => matches(kind, value)) && (typeof value !== 'string' || JSON.stringify(value) === text)) return value;
  if (kinds.includes('string')) return text;
  throw new Error('parameter differs from declared type');
}
function valid(schema: Schema, value: unknown): boolean {
  const supported = new Set(['type','properties','required','additionalProperties','items','enum','const','anyOf','oneOf','allOf','minimum','maximum','exclusiveMinimum','exclusiveMaximum','minLength','maxLength','pattern','minItems','maxItems','uniqueItems', ...annotation]);
  if (Object.keys(schema).some(k => !supported.has(k) && !k.startsWith('x-'))) return false;
  const kind = schema.type;
  if (kind !== undefined && !(Array.isArray(kind) ? kind : [kind]).some(t => typeof t === 'string' && matches(t,value))) return false;
  if (Array.isArray(schema.enum) && !schema.enum.some(x => JSON.stringify(x) === JSON.stringify(value))) return false;
  if ('const' in schema && JSON.stringify(schema.const) !== JSON.stringify(value)) return false;
  if (Array.isArray(schema.anyOf) && !schema.anyOf.some(s => valid(s as Schema,value))) return false;
  if (Array.isArray(schema.oneOf) && schema.oneOf.filter(s => valid(s as Schema,value)).length !== 1) return false;
  if (Array.isArray(schema.allOf) && !schema.allOf.every(s => valid(s as Schema,value))) return false;
  if (typeof value === 'number') {
    if (typeof schema.minimum === 'number' && value < schema.minimum || typeof schema.maximum === 'number' && value > schema.maximum ||
        typeof schema.exclusiveMinimum === 'number' && value <= schema.exclusiveMinimum || typeof schema.exclusiveMaximum === 'number' && value >= schema.exclusiveMaximum) return false;
  }
  if (typeof value === 'string') {
    if (typeof schema.minLength === 'number' && [...value].length < schema.minLength || typeof schema.maxLength === 'number' && [...value].length > schema.maxLength) return false;
    if (typeof schema.pattern === 'string' && !new RegExp(schema.pattern).test(value)) return false;
  }
  if (Array.isArray(value)) {
    if (typeof schema.minItems === 'number' && value.length < schema.minItems || typeof schema.maxItems === 'number' && value.length > schema.maxItems) return false;
    if (schema.uniqueItems === true && new Set(value.map(x => JSON.stringify(x))).size !== value.length) return false;
    if (schema.items && (typeof schema.items !== 'object' || !value.every(x => valid(schema.items as Schema,x)))) return false;
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>, props = (schema.properties ?? {}) as Record<string, Schema>;
    if (Array.isArray(schema.required) && schema.required.some(k => typeof k !== 'string' || !(k in obj))) return false;
    for (const [key,item] of Object.entries(obj)) {
      if (Object.hasOwn(props,key)) { if (!valid(props[key]!,item)) return false; }
      else if (schema.additionalProperties === false) return false;
      else if (schema.additionalProperties && typeof schema.additionalProperties === 'object' && !valid(schema.additionalProperties as Schema,item)) return false;
    }
  }
  return true;
}
/** No prose, partial frames, unoffered names or invalid argument schemas can become calls. */
export function parseNativeXmlCalls(text: string, offered: Tool[]): [string, Record<string, unknown>][] {
  const byName = new Map(offered.map(tool => [tool.function.name, tool.function.parameters]));
  if (byName.size !== offered.length) throw new Error('ambiguous offered tool names');
  let at = 0; const calls: [string, Record<string, unknown>][] = [];
  const ws = () => { while (at < text.length && /[ \t\r\n]/.test(text[at]!)) at++; };
  while (true) {
    ws(); if (at === text.length) break;
    if (!text.startsWith(OPEN, at)) throw new Error('mixed prose or missing tool frame');
    at += OPEN.length; ws();
    const nameMatch = /^<function=([^>\n]+)>\n/.exec(text.slice(at));
    if (!nameMatch) throw new Error('incomplete function frame');
    const name=nameMatch[1]!; at += nameMatch[0].length;
    const schema = byName.get(name);
    if (!schema || typeof schema !== 'object' || Array.isArray(schema)) throw new Error('tool not offered or missing schema');
    const props = ((schema as Schema).properties ?? {}) as Record<string, Schema>;
    const args: Record<string, unknown> = Object.create(null);
    for (;;) {
      const param = /^<parameter=([^>\n]+)>\n/.exec(text.slice(at));
      if (!param) break;
      const key=param[1]!; at += param[0].length;
      const end=text.indexOf(CLOSE,at);
      if (end < 0 || Object.hasOwn(args,key) || !Object.hasOwn(props,key)) throw new Error('incomplete, duplicate or unknown parameter');
      args[key]=valueFor(text.slice(at,end),props[key]!); at=end+CLOSE.length;
    }
    const tail=/^<\/function>\s*<\/tool_call>/.exec(text.slice(at));
    if (!tail) throw new Error('incomplete tool frame');
    at+=tail[0].length;
    if (!valid(schema as Schema,args)) throw new Error('tool arguments violate declared schema');
    calls.push([name, {...args}]);
  }
  if (!calls.length) throw new Error('no complete tool calls');
  return calls;
}
