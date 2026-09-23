/**
 * Program IR: a teacher task is a small natlang project (a root `.nl` function, its callable folder,
 * inputs, and the expected result), the same shape an application author writes. Producers that
 * describe functions as definitions convert them with `definitionProject`.
 */
import { natlangDefinition } from '../runtime/callable.js';
import { definitionNode } from '../runtime/kernel.js';
import { loadNamedFunction } from '../runtime/loader.js';
import { virtualSourceFiles } from '../runtime/virtual-project.js';
import { formatType, parseType } from '../native/types.js';
import type { LambdaNode } from '../native/values.js';

export const PROGRAM_VERSION = 'natlang.program/2';

export type ProgramSemantics = {
  /** Path of the root `.nl` function within `files`. */
  root: string;
  files: Record<string, string>;
  inputs: Record<string, unknown>;
  expected: unknown;
  operation?: string;
  effects?: Record<string, unknown>;
  failure_seed?: { code: string; kind?: 'compile' | 'runtime' | 'boundary' };
  folder_files?: Record<string, string>;
  expected_files?: Record<string, string>;
};
export type ProgramRecord = { version: string; id: string; kind: string; semantics: ProgramSemantics; [key: string]: unknown };

/** The root's kernel definition, loaded from the program's files. */
export function programDefinition(record: ProgramRecord) {
  return natlangDefinition(loadNamedFunction(`/project/${record.semantics.root}`, virtualSourceFiles(record.semantics.files)));
}

/** The pending root invocation of a program, with its inputs bound by parameter name. */
export function programNode(record: ProgramRecord): LambdaNode {
  const definition = programDefinition(record);
  const inputs = record.semantics.inputs ?? {};
  for (const name of Object.keys(inputs)) if (!definition.params.some(param => param.name === name))
    throw new TypeError(`${name} is not a parameter of ${definition.name}`);
  return definitionNode(definition, definition.params.map(param => inputs[param.name]));
}

/** A function described as data: natural-language `instructions`, or a TypeScript `code` body. */
export type FunctionSpec = { args?: Record<string, string>; returns: string; description?: string;
  instructions?: string; code?: string; async?: boolean; types?: Record<string, string>; kind?: string;
  codebase?: Record<string, FunctionSpec> };

const yamlText = (value: string) => JSON.stringify(value);
function natlangSource(spec: FunctionSpec): string {
  const args = Object.entries(spec.args ?? {});
  return `---\n${spec.description ? `description: ${yamlText(spec.description.trim())}\n` : ''}` +
    `args:${args.length ? '\n' + args.map(([name, type]) => `  ${name}: ${yamlText(type)}`).join('\n') : ' {}'}\n` +
    `returns: ${yamlText(spec.returns)}\n` +
    (spec.kind && spec.kind !== 'function' ? `kind: ${spec.kind}\n` : '') +
    (spec.types && Object.keys(spec.types).length ? `types:\n${Object.entries(spec.types).map(([name, type]) => `  ${name}: ${yamlText(type)}`).join('\n')}\n` : '') +
    `---\n${(spec.instructions ?? '').trim()}\n`;
}
function typeScriptSource(name: string, spec: FunctionSpec): string {
  // A module's children live in its companion folder and are imported explicitly.
  const children = Object.entries(spec.codebase ?? {}).map(([child, childSpec]) =>
    `import ${child} from './${name}/${child}.${childSpec.code !== undefined ? 'js' : 'nl'}';`);
  const params = Object.entries(spec.args ?? {}).map(([arg, type]) => `${arg.replace(/\?$/, '')}${arg.endsWith('?') ? '?' : ''}: ${type}`).join(', ');
  const body = (spec.code ?? '').trim();
  const isAsync = spec.async || /\bawait\b/.test(body);
  const imports = [...[...body.matchAll(/^import [^\n]+$/gm)].map(match => match[0]), ...children];
  const rest = body.replace(/^import [^\n]+\n?/gm, '');
  return `${imports.length ? imports.join('\n') + '\n\n' : ''}${spec.description ? `/** ${spec.description.trim().replace(/\s+/g, ' ')} */\n` : ''}` +
    `export default ${isAsync ? 'async ' : ''}function ${name}(${params}): ${isAsync ? `Promise<${spec.returns}>` : spec.returns} {\n` +
    `${rest.split('\n').map(line => line ? `  ${line}` : line).join('\n')}\n}\n`;
}

/**
 * The files of a project whose root is the natural-language function `name`. Child functions go in
 * its callable folder; named types shared with TypeScript children go in `name/types.ts`.
 */
export function definitionProject(name: string, root: FunctionSpec): { root: string; files: Record<string, string> } {
  const files: Record<string, string> = { [`${name}.nl`]: natlangSource(root) };
  const place = (dir: string, codebase: Record<string, FunctionSpec>) => {
    for (const [child, spec] of Object.entries(codebase)) {
      files[`${dir}/${child}.${spec.code !== undefined ? 'ts' : 'nl'}`] = spec.code !== undefined ? typeScriptSource(child, spec) : natlangSource(spec);
      if (spec.codebase && Object.keys(spec.codebase).length) place(`${dir}/${child}`, spec.codebase);
    }
  };
  place(name, root.codebase ?? {});
  const usesTypes = Object.values(files).some((text, index) => Object.keys(files)[index]!.endsWith('.ts'));
  if (usesTypes && root.types && Object.keys(root.types).length)
    files[`${name}/types.ts`] = Object.entries(root.types).map(([type, text]) => `export type ${type} = ${text};`).join('\n') + '\n';
  return { root: `${name}.nl`, files };
}

/** Split a lambda type `(a: A, b?: B) => R` into arguments and result, in natlang type syntax. */
export function lambdaSignature(type: string): { args: Record<string, string>; returns: string } {
  const parsed = parseType(type);
  if (parsed.kind !== 'lambda') throw new TypeError(`not a function type: ${type}`);
  return { args: Object.fromEntries(parsed.params.fields.map(field => [field.name + (field.optional ? '?' : ''), formatType(field.type)])),
    returns: formatType(parsed.returns) };
}
