/** Project scaffolding and saved input/output examples. No model calls are needed to create a program. */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync, renameSync } from 'node:fs';
import { basename, join, resolve, relative, sep } from 'node:path';
import { isDeepStrictEqual } from 'node:util';

export type SavedExample = { name?: string; function: string; inputs: Record<string, unknown>; expected: unknown };
export const sampleInput = 'I was charged twice for my subscription.';
const starterInstruction = 'Classify input as billing (charges, invoices or payments), technical (errors or bugs), or other. Return only the matching label.';

export function createProgram(directory: string, options: { instruction?: string; input?: string; returns?: string; profile?: string } = {}): string {
  const root = resolve(directory);
  if (existsSync(root) && (!statSync(root).isDirectory() || readdirSync(root).length)) throw new Error(`${root} is not empty. Choose a new directory; existing files are never overwritten.`);
  const name = basename(root).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+/, '') || 'my-program';
  const input = options.input ?? sampleInput;
  const returns = options.returns?.trim() || (options.instruction ? 'string' : '"billing" | "technical" | "other"');
  const instruction = options.instruction?.trim() || starterInstruction;
  const files: Record<string, string> = {
    'answer.nl': `---\ndescription: Answer using the program instructions.\nargs:\n  input: string\nreturns: ${JSON.stringify(returns)}\n---\n${instruction}\n`,
    'sample.json': JSON.stringify({ input }, null, 2) + '\n',
    'main.ts': `import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { TargetContext } from '@natlang/node';
import answer from './answer.nl';

export async function main(context: TargetContext): Promise<number> {
  const sample = JSON.parse(readFileSync(join(context.package?.root ?? context.workspace, 'sample.json'), 'utf8'));
  const input = (context.arguments.input as string | undefined) ?? sample.input;
  const result = await context.runtime.run(() => answer(input));
  context.io.output.write(JSON.stringify(result, null, 2) + '\\n');
  return 0;
}
`,
    'natlang.json': JSON.stringify({ schema: 'natlang.package/v2', name, version: '0.1.0', description: 'My first natlang program.',
      ...(options.profile ? { model: { profile: options.profile } } : {}),
      include: ['answer.nl', 'main.ts', 'sample.json', 'examples.json', 'package.json', 'tsconfig.json', 'README.md'],
      targets: { main: { entry: 'main.ts', description: 'Run with sample input, or supply your own text.',
        arguments: [{ name: 'input', type: 'string', description: 'Text to process; leave blank to use sample.json.' }] } },
      warmup: [{ name: 'sample input', function: 'answer.nl', inputs: { input } }] }, null, 2) + '\n',
    'package.json': JSON.stringify({ name, private: true, type: 'module' }, null, 2) + '\n',
    'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2023', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true,
      skipLibCheck: true, rootDir: '.', types: ['node'] }, include: ['*.ts'] }, null, 2) + '\n',
    'examples.json': JSON.stringify(options.instruction ? [] : [{ name: 'duplicate charge', function: 'answer.nl', inputs: { input }, expected: 'billing' }], null, 2) + '\n',
    '.gitignore': '.natlang/\nnode_modules/\n*.d.nl.ts\n',
    'README.md': `# ${name}\n\nRun these commands from this directory:\n\n\x60\x60\x60sh\nnatlang run\nnatlang run . -- "My own input"\nnatlang dev\nnatlang check\nnatlang test\nnatlang dashboard --open\n\x60\x60\x60\n\nEdit **answer.nl** to change the instructions, **sample.json** to change the default input, and **examples.json** to save input/expected-output tests. The dashboard can also save tests from the function form. Model tests execute real calls on your selected model.\n\nThe arguments and return type in answer.nl are the function's contract. Imports and call sites are checked before execution; the model's behavior still needs examples and tests. Instructions can use the declared input and callable helpers in the companion answer/ folder. Add typed services when the model needs outside capabilities.\n\nThe first run prepares your selected model. Local setup may download a model; you can instead configure an API provider in the dashboard's Models page. Runs reuse saved prompt openings automatically. Use a model profile to choose another backend.\n\nWhen a run fails, the dashboard shows its error and links calls to source and traces.\n`,
  };
  mkdirSync(root, { recursive: true });
  for (const [name, content] of Object.entries(files)) writeFileSync(join(root, name), content, { flag: 'wx' });
  return root;
}

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
