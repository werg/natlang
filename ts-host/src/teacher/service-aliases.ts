import ts from 'typescript';
import type { ExternalModule } from '../native/external.js';

export type ServiceAliasSpec = { source: string; methods: string[] };

/** Build capability namespaces from one already-instantiated host module, preserving its shared case state. */
export function bindServiceAliases(modules: Map<string, ExternalModule>, aliases: Record<string, ServiceAliasSpec>,
  services: Record<string, object>, declarations: Record<string, string>): void {
  const sources = new Set<string>();
  for (const [alias, spec] of Object.entries(aliases)) {
    const external = modules.get(spec.source);
    if (!external || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(alias) || !Array.isArray(spec.methods) || !spec.methods.length ||
      new Set(spec.methods).size !== spec.methods.length) throw new TypeError(`invalid service alias ${alias}`);
    for (const method of spec.methods) if (typeof external.exports[method] !== 'function')
      throw new TypeError(`service alias ${alias} names unavailable method ${method}`);
    const declarationFile = ts.createSourceFile(`${spec.source}.d.ts`, external.declaration, ts.ScriptTarget.ES2022, true);
    const moduleNode = declarationFile.statements.find(ts.isModuleDeclaration);
    const body = moduleNode?.body && ts.isModuleBlock(moduleNode.body) ? moduleNode.body : undefined;
    const localTypes = body?.statements.filter(statement => ts.isTypeAliasDeclaration(statement) ||
      ts.isInterfaceDeclaration(statement)).map(statement => statement.getFullText(declarationFile).trim()) ?? [];
    const signatures = body?.statements.filter(statement => ts.isFunctionDeclaration(statement) && statement.name &&
      spec.methods.includes(statement.name.text)).map(statement =>
      statement.getFullText(declarationFile).replace(/^\s*export\s+/, '').trim()) ?? [];
    if (signatures.length !== spec.methods.length) throw new TypeError(`service alias ${alias} declaration is incomplete`);
    services[alias] = Object.fromEntries(spec.methods.map(method => [method, external.exports[method]]));
    declarations[alias] = `declare namespace ${alias} {\n${[...localTypes, ...signatures]
      .map(signature => `  ${signature}`).join('\n')}\n}`;
    sources.add(spec.source);
  }
  // A split source module is an implementation detail. Leaving its original namespace visible would bypass every
  // alias filter and method scope. Retain it only when explicitly rebound as one of the restricted aliases.
  for (const source of sources) if (!Object.hasOwn(aliases, source)) {
    delete services[source]; delete declarations[source];
  }
}
