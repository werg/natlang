/**
 * The terminal's data files: the recipe catalog and its limits (`recipes.json`) and the fixed wording of the session's
 * messages (`messages.json`). An owner edits them without code; the loaders check their shape.
 */
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export type CommandRecipe = { id: string, description: string, argv: string[], cwd?: string, env?: Record<string, string>, timeoutMs?: number };
/** The limits a command runs under; each carries the reason it exists. */
export type RecipeLimits = { outputBytes: number, timeoutMs: number, maxTimeoutMs: number, killDelayMs: number };
export type RecipeData = { limits: RecipeLimits, recipes: CommandRecipe[] };
/** Message templates by name. `{name}` stands for a value the session supplies. */
export type MessageTable = Record<string, string>;

/** The data file `name`: beside this module (a package checkout) or in the source folder of the application (a build output). */
function dataFile(name: string): string {
  const path = [`./${name}`, `../../terminal/${name}`].map(candidate => fileURLToPath(new URL(candidate, import.meta.url))).find(existsSync);
  if (!path) throw new Error(`the terminal's data file ${name} is missing; it belongs next to natlang.json`);
  return path;
}

const LIMIT_NAMES = ['outputBytes', 'timeoutMs', 'maxTimeoutMs', 'killDelayMs'] as const;

/** Check and read `recipes.json` content: the limits (each `{ value, why }`) and the recipe list. */
export function parseRecipeData(source: unknown): RecipeData {
  const root = source as { limits?: Record<string, { value?: unknown }>, recipes?: unknown };
  if (!root || typeof root !== 'object' || !Array.isArray(root.recipes)) throw new Error('recipes.json holds { "limits": {...}, "recipes": [...] }');
  const limits = {} as RecipeLimits;
  for (const name of LIMIT_NAMES) {
    const value = root.limits?.[name]?.value;
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1)
      throw new Error(`recipes.json: limits.${name} is { "value": a positive whole number, "why": the reason }`);
    limits[name] = value;
  }
  return { limits, recipes: root.recipes as CommandRecipe[] };
}

export function loadRecipeData(path: string = dataFile('recipes.json')): RecipeData {
  return parseRecipeData(JSON.parse(readFileSync(path, 'utf8')));
}

export function loadMessages(path: string = dataFile('messages.json')): MessageTable {
  const table = JSON.parse(readFileSync(path, 'utf8')) as MessageTable;
  if (!table || typeof table !== 'object' || Object.values(table).some(value => typeof value !== 'string'))
    throw new Error('messages.json maps each message name to its text');
  return table;
}

/** The message `name` with each `{value}` filled in. A name or value the table does not have is reported with what the table offers. */
export function renderMessage(table: MessageTable, name: string, values: Record<string, string | number> = {}): string {
  const template = table[name];
  if (template === undefined) throw new Error(`no message named ${JSON.stringify(name)}; messages.json has ${Object.keys(table).join(', ')}`);
  return template.replace(/\{(\w+)\}/g, (_match, key: string) => {
    if (!(key in values)) throw new Error(`message ${JSON.stringify(name)} needs a value for {${key}}; the caller gave ${Object.keys(values).join(', ') || 'none'}`);
    return String(values[key]);
  });
}
