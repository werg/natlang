/** Project scaffolding and saved input/output examples. No model calls are needed to create a program. */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync, renameSync } from 'node:fs';
import { basename, join, resolve, relative, sep } from 'node:path';
import { isDeepStrictEqual } from 'node:util';

export type SavedExample = { name?: string; function: string; inputs: Record<string, unknown>; expected: unknown };
export const sampleInput = 'I was charged twice for my subscription.';
const starterInstruction = 'Classify input as billing (charges, invoices or payments), technical (errors or bugs), or other. Return only the matching label.';

/** A program's `answer.nl`: the starter's own unless the caller supplies instructions and a result type. */
export const answerSource = (instruction = starterInstruction, returns = '"billing" | "technical" | "other"') =>
  `---\ndescription: Answer using the program instructions.\nargs:\n  input: string\nreturns: ${JSON.stringify(returns)}\n---\n${instruction}\n`;

export function createProgram(directory: string, options: { instruction?: string; input?: string; returns?: string; profile?: string } = {}): string {
  const root = resolve(directory);
  if (existsSync(root) && (!statSync(root).isDirectory() || readdirSync(root).length)) throw new Error(`${root} is not empty. Choose a new directory; existing files are never overwritten.`);
  const name = basename(root).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+/, '') || 'my-program';
  const input = options.input ?? sampleInput;
  const returns = options.returns?.trim() || (options.instruction ? 'string' : '"billing" | "technical" | "other"');
  const instruction = options.instruction?.trim() || starterInstruction;
  // One function, its manifest and its tests: the package's main target is answer.nl itself, so there is no entry
  // module, tsconfig or input file to learn before the first edit. AGENTS.md (and CLAUDE.md, which imports it) tells a
  // coding agent the non-interactive loop.
  const files: Record<string, string> = {
    'answer.nl': answerSource(instruction, returns),
    'natlang.json': JSON.stringify({ schema: 'natlang.package/v2', name, version: '0.1.0', description: 'My first natlang program.',
      ...(options.profile ? { model: { profile: options.profile } } : {}),
      include: ['answer.nl', 'examples.json', 'README.md'],
      targets: { main: { entry: 'answer.nl', description: 'Run answer.nl on the sample input, or supply your own text.',
        arguments: [{ name: 'input', type: 'string', description: 'Text to process; leave blank for the sample.', default: input }] } },
      warmup: [{ name: 'sample input', function: 'answer.nl', inputs: { input } }] }, null, 2) + '\n',
    'examples.json': JSON.stringify(options.instruction ? [] : [{ name: 'duplicate charge', function: 'answer.nl', inputs: { input }, expected: 'billing' }], null, 2) + '\n',
    '.gitignore': '.natlang/\nnode_modules/\n*.d.nl.ts\n',
    'README.md': programReadme(name),
    'AGENTS.md': AGENT_GUIDE,
    'CLAUDE.md': '@AGENTS.md\n',
  };
  mkdirSync(root, { recursive: true });
  for (const [name, content] of Object.entries(files)) writeFileSync(join(root, name), content, { flag: 'wx' });
  return root;
}

const fence = '\x60\x60\x60';
const programReadme = (name: string) => `# ${name}

${fence}sh
natlang run                          # answer.nl on the sample input
natlang run . -- "My own input"      # or: natlang call answer.nl "My own input"
natlang dev                          # rerun after every edit; Ctrl+C stops
natlang test                         # check the saved examples
natlang dashboard --open             # results, traces and model settings in the browser
${fence}

**answer.nl** is a natural-language function: a typed signature (\`args\`, \`returns\`) and instructions that a model
carries out. Edit the instructions to change its behavior. **examples.json** saves inputs with their expected results, and
\`natlang test\` runs them on the real model. The sample input is the \`input\` argument's default in **natlang.json**.

\`natlang check\` checks the source without calling a model. The first run prepares the selected model: the local one is a
one-time download (about 1.8 GB), or choose a hosted model on the dashboard's Models page.
`;

/** The coding-agent guide every new program carries (AGENTS.md; CLAUDE.md imports it). */
export const AGENT_GUIDE = `# Working on this natlang program

natlang is TypeScript with natural-language functions. \`answer.nl\` is one: YAML front matter declares its typed
signature (\`args\`, \`returns\`), and the body is instructions that a model executes. The signature is checked like
TypeScript; the instructions' behavior is checked by running examples on the model.

## Files

- \`answer.nl\`: the function. Edit the instructions to change behavior, \`args\`/\`returns\` to change the contract.
- \`examples.json\`: saved cases, \`[{ name, function: "answer.nl", inputs: { input: ... }, expected }]\`, compared exactly.
- \`natlang.json\`: the package. \`targets.main\` runs \`answer.nl\`; its \`input\` argument's default is the sample.
- Generated (do not edit): \`*.d.nl.ts\` and \`.natlang/\`.

## The loop

1. \`natlang check --json\`: types, signatures and call sites, without a model. Fix every diagnostic before calling it.
2. \`natlang call answer.nl "some input" --json\`: one real call; arguments fill parameters in order.
3. \`natlang test --json\`: every example in examples.json; exits 1 if any fails, with expected and received values.
4. \`natlang traces list\` and \`natlang traces show CALL_ID\`: each step the model took, to see why a result is wrong.

Model calls take seconds on a hosted model and can take minutes on a local CPU, so check first and call deliberately.
When you change behavior, add or update an example and run \`natlang test\`.

## The model

If a command reports that the local runtime is not installed, ask the user before running \`natlang setup --yes\` (a
one-time download of about 1.8 GB), or use a hosted model: \`--provider ID\` (\`natlang models\` lists them) or
\`NATLANG_SERVER=URL NATLANG_MODEL=ID\` for an OpenAI-compatible endpoint. \`natlang doctor --json\` reports the setup.

## Writing natlang

- Exact work (arithmetic, parsing, sorting, I/O) belongs in TypeScript; judgment belongs in instructions.
- Put constraints in types: a string-literal union or a record in \`returns\` beats a sentence asking for a format.
- Helpers the function may call go in its companion folder, \`answer/\` (\`answer/lookup.ts\`, \`answer/summarize.nl\`).
- Number the steps of a multi-step instruction and name the data each step uses; the executing model may be small.
- Language reference and patterns: the natlang-authoring skill (\`natlang skills install\` adds it to Claude Code or Codex).
`;

export function readExamples(root: string): SavedExample[] {
  const path = join(root, 'examples.json');
  if (!existsSync(path)) return [];
  return validateExamples(root, JSON.parse(readFileSync(path, 'utf8')));
}

export function validateExamples(root: string, examples: unknown): SavedExample[] {
  const path = join(root, 'examples.json');
  if (!Array.isArray(examples)) throw new Error(`${path} must be an array of { function, inputs, expected } examples`);
  for (const item of examples) {
    if (!item || typeof item !== 'object' || typeof item.function !== 'string' || !item.function.endsWith('.nl') ||
        !item.inputs || typeof item.inputs !== 'object' || Array.isArray(item.inputs) || !Object.hasOwn(item, 'expected'))
      throw new Error(`${path}: each example needs a .nl function, an inputs object, and an expected result`);
    const delta = relative(resolve(root), resolve(root, item.function));
    if (delta === '..' || delta.startsWith('..' + sep) || resolve(root, item.function) === resolve(root))
      throw new Error('Example functions must be inside the program');
  }
  return examples;
}

export function saveExample(root: string, example: SavedExample): void {
  validateExamples(root, [example]);
  const examples = readExamples(root);
  // Re-saving an input updates its expectation rather than growing duplicate tests.
  const previous = examples.findIndex(item => item.function === example.function && isDeepStrictEqual(item.inputs, example.inputs));
  if (previous >= 0) examples[previous] = { ...examples[previous]!, ...example }; else examples.push(example);
  const path = join(root, 'examples.json'), temporary = path + '.tmp';
  writeFileSync(temporary, JSON.stringify(examples, null, 2) + '\n');
  renameSync(temporary, path);
}
