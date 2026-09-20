export const crisp = ({ id, name, category, level, description, concepts, root, args = {}, returns,
  code, files = {}, inputs = {}, expected }) => ({
  id, name, category, level, description, concepts, modelRequired: false, root,
  files: { [root]: `/*---\ndescription: ${description}\n${Object.keys(args).length ?
    `args:\n${Object.entries(args).map(([key, type]) => `  ${key}: ${type}`).join('\n')}\n` : ''}returns: ${returns}\nengine: typescript-host\n---*/\n${code.trim()}\n`, ...files },
  inputs, expected,
});

export const natural = ({ id, name, category = 'Natural language', level, description, concepts,
  root, source, files = {}, inputs = {}, expected }) => ({
  id, name, category, level, description, concepts, modelRequired: true, root,
  files: { [root]: source.trim() + '\n', ...files }, inputs, expected,
});
