/** Exact complete LFM3/Qwen XML calls, using the same first-close-delimiter law as serve/chat.py. */
type Schema = Record<string, unknown>;
type Tool = { function: { name: string; parameters?: unknown } };
const OPEN = '<tool_call>';
const CLOSE = '\n</parameter>\n';
const annotation = new Set(['description', 'title', 'default', 'examples', '$schema', 'deprecated', 'readOnly', 'writeOnly']);

function resolveRef(ref: unknown, root: Schema): Schema | boolean {
  if (typeof ref !== 'string' || !ref.startsWith('#/')) throw new Error('unsupported schema reference');
  let node: unknown = root;
  for (const part of ref.slice(2).split('/')) {
    const key=part.replace(/~1/g,'/').replace(/~0/g,'~');
    if (!node || typeof node !== 'object' || Array.isArray(node) || !Object.hasOwn(node,key)) throw new Error('unresolved schema reference');
    node=(node as Schema)[key];
  }
  if (typeof node !== 'boolean' && (!node || typeof node !== 'object' || Array.isArray(node))) throw new Error('invalid schema reference');
  return node as Schema | boolean;
}
function equal(a: unknown,b: unknown): boolean {
  if (a===b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length===b.length && a.every((v,i)=>equal(v,b[i]));
  if (a && b && typeof a==='object' && typeof b==='object' && !Array.isArray(a) && !Array.isArray(b)) {
    const x=a as Schema,y=b as Schema, keys=Object.keys(x);
    return keys.length===Object.keys(y).length && keys.every(k=>Object.hasOwn(y,k) && equal(x[k],y[k]));
  }
  return false;
}
function types(schema: Schema, root: Schema = schema, depth=0): string[] | undefined {
  if (depth>64) throw new Error('recursive schema reference');
  if (schema.$ref !== undefined) { const target=resolveRef(schema.$ref,root); if(typeof target==='object') return types(target,root,depth+1); return undefined; }
  if (typeof schema.type === 'string') return [schema.type];
  if (Array.isArray(schema.type) && schema.type.every(x => typeof x === 'string')) return schema.type as string[];
  for (const name of ['anyOf', 'oneOf']) if (Array.isArray(schema[name])) {
    const parts = (schema[name] as Schema[]).map(part=>types(part,root,depth+1));
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
function renderedJson(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(renderedJson).join(', ') + ']';
  if (value && typeof value === 'object') return '{' + Object.entries(value as Record<string, unknown>).map(([k,v]) => JSON.stringify(k) + ': ' + renderedJson(v)).join(', ') + '}';
  return JSON.stringify(value);
}
function valueFor(text: string, schema: Schema, root: Schema): unknown {
  const kinds = types(schema,root);
  if (kinds?.length === 1 && kinds[0] === 'string') return text;
  let value: unknown;
  try { value = JSON.parse(text); }
  catch { if (kinds && !kinds.includes('string')) throw new Error('parameter is not declared JSON type'); return text; }
  if (!kinds) return typeof value === 'string' ? (JSON.stringify(value) === text ? value : text) : (renderedJson(value) === text ? value : text);
  if (kinds.some(kind => matches(kind, value)) && (typeof value !== 'string' || JSON.stringify(value) === text)) return value;
  if (kinds.includes('string')) return text;
  throw new Error('parameter differs from declared type');
}
function valid(schema: Schema | boolean, value: unknown, root: Schema, depth=0): boolean {
  if (depth>64) return false;
  if (typeof schema==='boolean') return schema;
  if (!schema || typeof schema!=='object' || Array.isArray(schema)) return false;
  const supported = new Set(['type','properties','required','additionalProperties','items','enum','const','anyOf','oneOf','allOf','minimum','maximum','exclusiveMinimum','exclusiveMaximum','minLength','maxLength','pattern','minItems','maxItems','uniqueItems','$ref','$defs','definitions', ...annotation]);
  if (Object.keys(schema).some(k => !supported.has(k) && !k.startsWith('x-'))) return false;
  if (schema.$ref !== undefined && !valid(resolveRef(schema.$ref,root),value,root,depth+1)) return false;
  const kind = schema.type;
  if (kind !== undefined && !(Array.isArray(kind) ? kind : [kind]).some(t => typeof t === 'string' && matches(t,value))) return false;
  if (Array.isArray(schema.enum) && !schema.enum.some(x => equal(x,value))) return false;
  if ('const' in schema && !equal(schema.const,value)) return false;
  if (Array.isArray(schema.anyOf) && !schema.anyOf.some(s => valid(s as Schema,value,root,depth+1))) return false;
  if (Array.isArray(schema.oneOf) && schema.oneOf.filter(s => valid(s as Schema,value,root,depth+1)).length !== 1) return false;
  if (Array.isArray(schema.allOf) && !schema.allOf.every(s => valid(s as Schema,value,root,depth+1))) return false;
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
    if (schema.uniqueItems === true && value.some((x,i)=>value.slice(0,i).some(y=>equal(x,y)))) return false;
    if (schema.items !== undefined && !value.every(x => valid(schema.items as Schema | boolean,x,root,depth+1))) return false;
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>, props = (schema.properties ?? {}) as Record<string, Schema>;
    if (Array.isArray(schema.required) && schema.required.some(k => typeof k !== 'string' || !(k in obj))) return false;
    for (const [key,item] of Object.entries(obj)) {
      if (Object.hasOwn(props,key)) { if (!valid(props[key]!,item,root,depth+1)) return false; }
      else if (schema.additionalProperties === false) return false;
      else if (schema.additionalProperties && typeof schema.additionalProperties === 'object' && !valid(schema.additionalProperties as Schema,item,root,depth+1)) return false;
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
    let argumentSchema=schema as Schema;
    for (let depth=0; argumentSchema.$ref !== undefined; depth++) {
      if (depth>64) throw new Error('recursive root schema reference');
      const target=resolveRef(argumentSchema.$ref,schema as Schema);
      if (typeof target !== 'object') throw new Error('tool root reference is not an object schema');
      const {$ref: _ref,...siblings}=argumentSchema;argumentSchema={...target,...siblings};
    }
    const props = (argumentSchema.properties ?? {}) as Record<string, Schema>;
    const args: Record<string, unknown> = Object.create(null);
    for (;;) {
      const param = /^<parameter=([^>\n]+)>\n/.exec(text.slice(at));
      if (!param) break;
      const key=param[1]!; at += param[0].length;
      const end=text.indexOf(CLOSE,at);
      if (end < 0 || Object.hasOwn(args,key)) throw new Error('incomplete or duplicate parameter');
      const parameterSchema = Object.hasOwn(props,key) ? props[key] : argumentSchema.additionalProperties;
      if (parameterSchema === false) throw new Error('unknown parameter');
      args[key]=valueFor(text.slice(at,end), parameterSchema && typeof parameterSchema === 'object' ? parameterSchema as Schema : {},schema as Schema); at=end+CLOSE.length;
    }
    const tail=/^<\/function>\s*<\/tool_call>/.exec(text.slice(at));
    if (!tail) throw new Error('incomplete tool frame');
    at+=tail[0].length;
    if (!valid(schema as Schema,args,schema as Schema)) throw new Error('tool arguments violate declared schema');
    calls.push([name, {...args}]);
  }
  if (!calls.length) throw new Error('no complete tool calls');
  return calls;
}
