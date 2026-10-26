/**
 * Compile the lexical adapter-scope tree of a program (plans/HIERARCHICAL_LORA_TUNING.md §3.1).
 *
 * The context DAG has no unique parent, so the tree is declared over bound definitions: the program root, one folder
 * node per directory on a definition's source path (author-declared tree roots become `tree` nodes), and one lambda
 * node per named definition or persistent inline site. Two aliases of one bound callable resolve to the same binding
 * because resolution keys on the callee's definition, not on the path a caller took.
 */
import type { ProgramDescriptor } from '../types.js';
import { fingerprint } from '../identity.js';
import type { ScopeNode, WeightBindingEntry } from './manifest.js';

export type ScopeOptions = {
  /** Directories (relative to the program root, `/`-separated) that root their own tree level. */
  readonly treeRoots?: readonly string[];
};

export const ROOT_NODE = 'tree:';
const folderNode = (path: string, trees: ReadonlySet<string>) => (trees.has(path) ? 'tree:' : 'folder:') + path;

/** The semantic hash a binding's adapters were trained against: its baseline instructions and frozen contract. */
export function bindingSemanticHash(component: ProgramDescriptor['components'][number]): string {
  return fingerprint({ baseline: component.baselineHash, contract: component.contractHash }, 'natlang.weights-semantic/v1');
}

export function compileScopeTree(program: ProgramDescriptor, options: ScopeOptions = {}):
  { tree: ScopeNode[]; bindings: WeightBindingEntry[] } {
  const trees = new Set((options.treeRoots ?? []).map(path => path.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '')));
  for (const root of trees) if (!root || root.split('/').includes('..')) throw new Error(`tree root ${JSON.stringify(root)} must be a directory inside the program`);
  const nodes = new Map<string, ScopeNode>([[ROOT_NODE, { id: ROOT_NODE, level: 'tree', path: '', parent: null }]]);
  const bindings: WeightBindingEntry[] = [];
  for (const component of program.components) {
    if (component.origin === 'program' || !component.source) continue;
    const segments = component.source.path.replace(/\\/g, '/').split('/').filter(Boolean);
    // A named definition's own companion folder holds its children; the file itself is the lambda.
    const directories = segments.slice(0, -1);
    const ancestry = [ROOT_NODE];
    for (let index = 1; index <= directories.length; index++) {
      const path = directories.slice(0, index).join('/');
      const id = folderNode(path, trees);
      if (!nodes.has(id)) nodes.set(id, { id, level: trees.has(path) ? 'tree' : 'folder', path, parent: ancestry.at(-1)! });
      ancestry.push(id);
    }
    const lambda = 'lambda:' + component.key;
    nodes.set(lambda, { id: lambda, level: 'lambda', path: component.source.path, parent: ancestry.at(-1)! });
    ancestry.push(lambda);
    bindings.push({ id: fingerprint({ program: program.id, component: component.key, context: null }, 'natlang.weights-binding/v1').slice(0, 24),
      component: component.key, definitionId: component.definitionId ?? null, contextId: null,
      origin: component.origin, ancestry, semanticHash: bindingSemanticHash(component) });
  }
  for (const root of trees) if (![...nodes.values()].some(node => node.level === 'tree' && node.path === root))
    throw new Error(`tree root ${root} contains no bound definition`);
  return { tree: [...nodes.values()].sort((a, b) => a.id.localeCompare(b.id)), bindings: bindings.sort((a, b) => a.component.localeCompare(b.component)) };
}
