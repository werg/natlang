export const typescript = (name, args, returns, code) => {
  const parameters = Object.entries(args).map(([raw, type]) =>
    `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${type}`).join(', ');
  const body = code.trim();
  const async = /\bawait\b/.test(body);
  return `export default ${async ? 'async ' : ''}function ${name}(${parameters}): ` +
    `${async ? `Promise<${returns}>` : returns} {\n${body}\n}\n`;
};

export const crisp = ({ id, name, category, level, description, concepts, root, args = {}, returns,
  code, files = {}, inputs = {}, expected }) => {
  const slash = root.lastIndexOf('/'), directory = slash < 0 ? '' : root.slice(0, slash + 1);
  const typePath = `${directory}types.ts`, sourceFiles = { ...files };
  let module = typescript(root.split('/').at(-1).replace(/\.ts$/, ''), args, returns, code);
  if (typeof sourceFiles[typePath] === 'string') {
    sourceFiles[typePath] = sourceFiles[typePath].replace(/^(?!export\s+)type\s+/gm, 'export type ');
    const names = [...sourceFiles[typePath].matchAll(/^export\s+type\s+([A-Za-z_$][\w$]*)/gm)].map(match => match[1]);
    if (names.length) module = `import type { ${names.join(', ')} } from "./types.js";\n\n${module}`;
  }
  return { id, name, category, level, description, concepts, modelRequired: false, root,
    files: { [root]: module, ...sourceFiles }, inputs, expected };
};

export const natural = ({ id, name, category = 'Natural language', level, description, concepts,
  root, source, files = {}, inputs = {}, expected }) => ({
  id, name, category, level, description, concepts, modelRequired: true, root,
  files: { [root]: source.trim() + '\n', ...files }, inputs, expected,
});
