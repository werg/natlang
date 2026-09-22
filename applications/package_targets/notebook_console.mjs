import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { NotebookWorkspace } from '../notebook.mjs';
import { flag, terminalExecutable } from './terminal_helpers.mjs';

export const STARTER_NOTEBOOK = {
  tables: { orders: [{ region: 'north', amount: 12 }, { region: 'south', amount: 8 },
    { region: 'north', amount: 5 }] },
  cells: [
    { id: 'regional_totals', engine: 'sqlite', needs: [], description: 'total order amount by region',
      source: 'SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region' },
    { id: 'largest_region', engine: 'typescript-host', needs: ['regional_totals'],
      description: 'identify the region with the largest total',
      source: 'return args.deps.regional_totals.toSorted((a, b) => b.total - a.total)[0];' },
  ],
};

export function createTarget(context) {
  const path = flag(context.args, '--notebook');
  const config = path ? JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8')) : STARTER_NOTEBOOK;
  const Environment = context.runtime.TypeScriptEnvironment;
  const notebook = new NotebookWorkspace(config.cells, config.tables ?? {},
    { environment: new Environment({ mode: 'fresh' }) });
  return terminalExecutable(context, { hostObject: { notebook, drainEvents: () => notebook.drainEvents() },
    reducerInputs: () => ({ files: new context.runtime.NodeFileTree(context.workspace) }),
    initialState: () => ({ requests: [], runs: [], status: 'idle' }),
    event: (value, id) => ({ id, kind: 'request', value }),
    commands: {
      cells: { description: 'list notebook cells and dependencies', run: () => notebook.catalog()
        .map(cell => `${cell.id}  [${cell.engine}]  needs: ${cell.needs.join(', ') || 'none'}  ${cell.description}`).join('\n') },
      load: { description: 'FILE import or replace cells and tables', run: value => {
        if (!value) throw new Error('provide a notebook JSON path');
        const imported = notebook.importConfig(JSON.parse(readFileSync(resolve(context.workspace, value), 'utf8')));
        return `Loaded ${imported.cells.length} cells and ${imported.tables.length} tables.`;
      } },
      example: { description: 'show a useful first request', run: () =>
        'Which region has the largest total, and what is that total?' },
    }, close: () => notebook.close() });
}
