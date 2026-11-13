/** A program's examples.json: the saved { function, inputs, expected? } cases `natlang test` checks, `natlang tune`
 * learns from and feedback (`natlang traces good|bad|fix`, the dashboard, escalations) adds to. One reader and one writer. */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';

/** One entry of a program's examples.json: the file `natlang test` checks, `natlang tune` learns from and feedback
 * (`natlang traces good|bad|fix`, the dashboard, escalations) adds to. An entry without `expected` is an input only. */
export type SavedExample = { name?: string; function: string; inputs: Record<string, unknown>; expected?: unknown; group?: string;
  producer?: string; protected?: boolean; [key: string]: unknown };

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
        !item.inputs || typeof item.inputs !== 'object' || Array.isArray(item.inputs))
      throw new Error(`${path}: each example needs a .nl function and an inputs object (and an expected result to be tested)`);
    const delta = relative(resolve(root), resolve(root, item.function));
    if (delta === '..' || delta.startsWith('..' + sep) || resolve(root, item.function) === resolve(root))
      throw new Error('Example functions must be inside the program');
  }
  return examples;
}

/** Key-sorted JSON: equal inputs compare equal whatever order their keys were written in. */
const stable = (value: unknown): string => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v) ?
  Object.fromEntries(Object.keys(v).sort().map(key => [key, (v as Record<string, unknown>)[key]])) : v);

/**
 * Add `example` to the program's examples.json, or replace the expected result of the entry with the same function and
 * inputs (atomic rewrite). An input alone never changes a listed entry; a protected entry is held for final reporting
 * and never changed.
 */
export function saveExample(root: string, example: SavedExample): 'added' | 'updated' | 'unchanged' {
  validateExamples(root, [example]);
  const examples = existsSync(join(root, 'examples.json')) ? readExamples(root) : [];
  const key = stable([example.function, example.inputs]);
  const index = examples.findIndex(item => stable([item.function, item.inputs]) === key);
  const found = index >= 0 ? examples[index]! : undefined;
  const path = join(root, 'examples.json');
  if (found) {
    if (!('expected' in example)) return 'unchanged';
    if (found.protected === true) throw new Error(`${path} holds this input as a protected example; it is kept unchanged`);
    if ('expected' in found && stable(found.expected) === stable(example.expected)) return 'unchanged';
    examples[index] = { ...found, expected: example.expected, ...('producer' in example ? { producer: example.producer } : {}) };
  } else examples.push(example);
  mkdirSync(root, { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(examples, null, 2) + '\n');
  renameSync(temporary, path);
  return found ? 'updated' : 'added';
}
