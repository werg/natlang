const typeNames = source => [...String(source ?? '').matchAll(/^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)/gm)].map(match => match[1]);

/**
 * A TypeScript entry module: `main(input)` receives the named inputs as one record. Types come from the
 * sibling `types.ts` (pass its source to import the names it declares).
 */
export const typescript = (name, args, returns, code, types) => {
  const fields = Object.entries(args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${type}`);
  const names = Object.keys(args).map(raw => raw.replace(/\?$/, ''));
  const body = code.trim().split('\n').map(line => line ? `  ${line}` : line).join('\n');
  const async = /\bawait\b/.test(body);
  const imported = typeNames(types);
  return `${imported.length ? `import type { ${imported.join(', ')} } from './types.js';\n\n` : ''}` +
    `/** ${name} */\nexport ${async ? 'async ' : ''}function main(input: { ${fields.join(', ')} }): ` +
    `${async ? `Promise<${returns}>` : returns} {\n${names.length ? `  const { ${names.join(', ')} } = input;\n` : ''}${body}\n}\n`;
};

const exported = source => source.replace(/^(?!export\s+)type\s+/gm, 'export type ');

export const crisp = ({ id, name, category, level, description, concepts, root, args = {}, returns,
  code, files = {}, inputs = {}, expected }) => {
  const slash = root.lastIndexOf('/'), directory = slash < 0 ? '' : root.slice(0, slash + 1);
  const typePath = `${directory}types.ts`, sourceFiles = { ...files };
  if (typeof sourceFiles[typePath] === 'string') sourceFiles[typePath] = exported(sourceFiles[typePath]);
  const module = typescript(root.split('/').at(-1).replace(/\.ts$/, ''), args, returns, code, sourceFiles[typePath]);
  return { id, name, category, level, description, concepts, modelRequired: false, root,
    files: { [root]: module, ...sourceFiles }, inputs, expected };
};

export const natural = ({ id, name, category = 'Natural language', level, description, concepts,
  root, source, files = {}, inputs = {}, expected }) => {
  const slash = root.lastIndexOf('/'), typePath = `${slash < 0 ? '' : root.slice(0, slash + 1)}types.ts`, sourceFiles = { ...files };
  if (typeof sourceFiles[typePath] === 'string') sourceFiles[typePath] = exported(sourceFiles[typePath]);
  return { id, name, category, level, description, concepts, modelRequired: true, root,
    files: { [root]: source.trim() + '\n', ...sourceFiles }, inputs, expected };
};

/** A function in a natural-language function's companion folder: a default export with typed parameters. */
export const helper = (name, args, returns, code) => {
  const parameters = Object.entries(args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${type}`).join(', ');
  const body = code.trim();
  const async = /\bawait\b/.test(body);
  return `export default ${async ? 'async ' : ''}function ${name}(${parameters}): ${async ? `Promise<${returns}>` : returns} {\n${body}\n}\n`;
};
