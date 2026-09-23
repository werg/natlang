import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { openAICompatibleModelTurn } from '../ts-host/dist/index.js';

export function cliFlag(args, name, fallback = null) {
  const at = args.indexOf(name);
  return at < 0 ? fallback : args[at + 1] ?? null;
}

export function modelTurnFromCli(args) {
  const endpoint = cliFlag(args, '--server', process.env.NATLANG_SERVER);
  const model = cliFlag(args, '--model', process.env.NATLANG_MODEL);
  if (!endpoint || !model) throw new Error('use --server URL --model ID (or NATLANG_SERVER/NATLANG_MODEL)');
  const exchangePath = cliFlag(args, '--exchanges');
  return openAICompatibleModelTurn({ endpoint, model,
    apiKey: process.env.NATLANG_API_KEY,
    onExchange: exchangePath ? exchange => {
      const path = resolve(exchangePath); mkdirSync(dirname(path), { recursive: true });
      appendFileSync(path, JSON.stringify(exchange) + '\n');
    } : undefined });
}
